import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(3,3), padding_size=(1,1)):
        super(Conv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, padding=1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size,3, 1, padding=1),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Up_concat(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up_concat, self).__init__()
        self.conv = Conv(in_size + out_size, out_size)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        #         #paper里面说，这里不需要padding，可以尝试去掉这个部分
        #         offset = outputs2.size()[2] - inputs1.size()[2]
        #         padding = 2 * [offset // 2, offset // 2, 0]
        #         outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([inputs1, outputs2], 1))