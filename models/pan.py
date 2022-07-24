import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):

    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x

def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)

    return x

def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class PyramidAttention(nn.Module):
    def __init__(self, res_scale=1, channel=64, channels =64, reduction=2, kernelsize=1, ksize=3, stride=1, softmax_scale=10, conv=default_conv):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.escape_NaN = torch.FloatTensor([1e-4])
        self.conv_match_L_base = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(conv,channels,channel//reduction,1,bn=False,act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channels, channel, 1, bn=False, act=nn.PReLU())



    def forward(self, input_l, input_s):
        res = input_l
        N, C_l, H_l, W_l = input_l.size()#n,16,h,w
        N, C_s, H_s, W_s = input_s.size()#n,12,h,w
        batch_size = N

        #theta
        match_base = self.conv_match_L_base(input_l)#n,8,h,w
        shape_base = list(res.size())#n,16,h,w
        input_groups = torch.split(match_base, 1, dim=0)

        #patch size for matching
        kernel = self.ksize #3

        #raw_w is for reconstruction
        raw_w = []
        #w is for matching
        w = []

        #conv_f -input:input_s
        base = self.conv_assembly(input_s)
        shape_input = base.shape
        #sampling
        raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                        strides=[self.stride, self.stride],
                                        rates=[1, 1],
                                        padding='same') #N, C*k*k, L
        raw_w_i = raw_w_i.view(batch_size, shape_input[1], kernel, kernel, H_s * W_s)
        raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3) #raw_shape:[N,L,C,k,k]
        raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
        raw_w.append(raw_w_i_groups)

        #conv_g -input:input_s
        ref_i = self.conv_match(input_s)
        shape_ref = ref_i.shape
        #sampling
        w_i = extract_image_patches(ref_i, ksizes=[kernel, kernel],
                                    strides=[self.stride, self.stride],
                                    rates=[1, 1],
                                    padding='same')
        w_i = w_i.view(shape_ref[0], shape_ref[1], kernel, kernel, -1)
        w_i = w_i.permute(0, 4, 1, 2, 3) #w shape:[N, L, C, k, k]
        w_i_groups = torch.split(w_i, 1, dim=0)
        w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            wi = w[0][idx][0] #H_s*W_s, channels//reduction,3,3  64,32,3,3
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                           axis=[1, 2, 3],
                                           keepdim=True)) + self.escape_NaN
            wi_normed = wi/max_wi
            #matching
            xi = same_padding(xi, [kernel, kernel], [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1)
            yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3])

            #softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)

            raw_wi = raw_w[0][idx][0]
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride, padding=1)/4.
            y.append(yi)

        y = torch.cat(y, dim=0) + res*self.res_scale
        return y

######################################################################################################################################################################################################################
def get_kernel():
    k = np.float32([.0625, .25, .375, .25, .0625])
    k = np.outer(k, k)
    return k

#自定义卷积核进行卷积操作
class GaussianPyramid(nn.Module):
    def __init__(self, n):
        super(GaussianPyramid, self).__init__()
        kernel=get_kernel()
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) #(H,W,1,1)
        self.n=n
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, img):
        levels=[]
        levels.append(img)
        low=img
        for i in range(self.n - 1):
            low = F.conv2d(low, self.weight, stride=(2,2),padding=(2,2))
            levels.append(low)
        return levels[::-1]
    #return 高斯金字塔


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()

        x_phi = self.conv_phi(x).view(b, c, -1)
        x_theta = self.conv_theta(x).view(b, c, -1)
        x_g = self.conv_g(x).view(b, c, -1)


        x_phi1 = x_phi.view(b, -1)
        x_phi1 = self.softmax(x_phi1)
        x_phi1 = x_phi1.view(b, c, -1)#(N,C,HW)

        x_g1 = x_g.view(b, -1)
        x_g1 = self.softmax(x_g1)
        x_g1 = x_g1.view(b, c, -1) #(N,C,HW)

        y = torch.matmul(x_g1, x_phi1.permute(0, 2, 1).contiguous()) #(N,C,C)
        y = torch.matmul(y, x_theta) #(N, C, HW)

        F_s = y.view(b, self.inter_channel, h, w)
        spatial_out = self.conv_mask(F_s)
        return spatial_out


class subnet(nn.Module):
    def __init__(self, in_channel, num, nums, kernelsize=3, Cross=False):
        super(subnet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channel, out_channels=num, kernel_size=kernelsize, padding=1)
        self.non_local1 = NonLocalBlock(num)
        self.conv1 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=3, padding=3)
        self.pyramid = PyramidAttention(channel=num, channels=nums)
        self.conv4 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=3, padding=3)
        self.conv5 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, dilation=2, padding=2)
        self.conv6 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=kernelsize, padding=1)
        self.non_local2 = NonLocalBlock(num)
        self.recon = nn.Conv2d(in_channels=num, out_channels=8, kernel_size=kernelsize, padding=1)
        self.cross = Cross

    def forward(self, MS, PAN, small=None):
        images = torch.cat((MS,PAN),1)
        x0 = self.conv0(images)
        x0 = x0 + self.non_local1(x0)
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        middle = x3
        if self.cross:
            middle = self.pyramid(middle, small)

        x4 = F.relu(self.conv4(middle+x3))
        x5 = F.relu(self.conv4(x4+x2))
        x6 = F.relu(self.conv4(x5 + x1))
        x7 = x6 + self.non_local2(x6)
        x7 = MS + self.recon(x7+x0)
        out = x7
        return out, middle, x7

class LPNet(nn.Module):
    def __init__(self, num1, num2, num3):
        super(LPNet, self).__init__()
        self.subnet1 = subnet(in_channel=9, num=num1, nums=num1, Cross=False)
        self.subnet2 = subnet(in_channel=9, num=num2, nums = num1, Cross=True)
        self.subnet3 = subnet(in_channel=9, num=num3, nums = num2, Cross=True)
        self.gaussian = GaussianPyramid(3)

    def forward(self, MS, PAN):
        pyramid = self.gaussian(PAN)
        out1, global_1, outf1 = self.subnet1(MS,pyramid[0])
        out1_t = F.interpolate(outf1, size=[(pyramid[1].shape)[2], (pyramid[1].shape)[3]], mode='nearest')
        out2, global_2, outf2 = self.subnet2(out1_t, pyramid[1], small=global_1)
        out2_t = F.interpolate(outf2, size=[(pyramid[2].shape)[2], (pyramid[2].shape)[3]], mode='nearest')
        out3,_,_ = self.subnet3(out2_t, pyramid[2], small=global_2)
        output_pyramid = []
        output_pyramid.append(out1)
        output_pyramid.append(out2)
        output_pyramid.append(out3)
        return output_pyramid

     ############################################test###############################################################

if __name__ == "__main__":
    
    MS = torch.ones([16, 8, 4, 4])
    PAN = torch.ones([16, 1, 16, 16])
    lpnet = LPNet(12, 16, 24)
    y = lpnet(MS, PAN) #list
    print(type(y), len(y))
    print(y[2].shape)