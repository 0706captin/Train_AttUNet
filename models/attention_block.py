import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention_block(nn.Module):
    def __init__(self, in_c, g_c, inter_c):
        super(Attention_block, self).__init__()

        # x->thrta_x
        self.theta = nn.Conv2d(in_channels=in_c, out_channels=inter_c,
                               kernel_size=2, stride=2, padding=0, bias=False)

        # gating -> phi_g
        self.phi = nn.Conv2d(in_channels=g_c, out_channels=inter_c,
                             kernel_size=1, stride=1, padding=0, bias=True)

        # f1->f2
        self.psi = nn.Conv2d(in_channels=inter_c, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # y->g_conv(output)
        self.W = nn.Sequential(nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1, stride=1, padding=0),
                               nn.BatchNorm2d(in_c)
                               )

    def forward(self, x, g):
        input_size = x.size()
        #         print('input_size:',x.shape)

        # x->thrta_x
        theta_x = self.theta(x)
        #         print('theta_x:',theta_x.shape)
        theta_x_size = theta_x.size()  #####?????????????

        # gating -> phi_g
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)

        # f1->f2
        f1 = F.relu(theta_x + phi_g, inplace=True)
        f2 = self.psi(f1)

        # ->α

        sigm_f2 = torch.sigmoid(f2)
        a = F.interpolate(sigm_f2, size=input_size[2:], mode='bilinear', align_corners=True)
        y = a.expand_as(x) * x
        output = self.W(y)
        return output
