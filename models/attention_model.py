from models.attention_block import *
from models.unet_parts import *


class Unet_att(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Unet_att, self).__init__()
        # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16, 16, 3, 1, 1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(inplace=True)
                                )
        self.m1 = nn.MaxPool2d(2)

        self.c2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True)
                                )
        self.m2 = nn.MaxPool2d(2)

        self.c3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True)
                                )
        self.m3 = nn.MaxPool2d(2)

        self.c4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128, 128, 3, 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True)
                                )
        self.m4 = nn.MaxPool2d(2)

        self.c5 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, 1, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True)
                                )
        self.m5 = nn.MaxPool2d(2)

        self.cen = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, 512, 3, 1, 1),
                                 nn.BatchNorm2d(512),
                                 nn.ReLU(inplace=True)
                                 )

        self.gating_signal = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True))

        self.attentionblock5 = Attention_block(256, 256, 256)
        self.attentionblock4 = Attention_block(128, 256, 128)
        self.attentionblock3 = Attention_block(64, 256, 64)
        self.attentionblock2 = Attention_block(32, 256, 32)

        self.up_concat5 = Up_concat(512, 256)
        self.up_concat4 = Up_concat(256, 128)
        self.up_concat3 = Up_concat(128, 64)
        self.up_concat2 = Up_concat(64, 32)
        self.up_concat1 = Up_concat(32, 16)

        self.final = nn.Conv2d(16, n_classes, 1)

    def forward(self, input):
        ########### fearure extraction(Encoder) #############
        conv1 = self.c1(input)

        max1 = self.m1(conv1)
        #         print("conv1:",conv1.shape)
        #         print("max1:",max1.shape)
        conv2 = self.c2(max1)
        max2 = self.m2(conv2)
        #         print("conv2:",conv2.shape)
        #         print("max2:",max2.shape)
        conv3 = self.c3(max2)
        max3 = self.m3(conv3)
        #         print("conv3:",conv3.shape)
        #         print("max3:",max3.shape)
        conv4 = self.c4(max3)
        max4 = self.m4(conv4)
        #         print("conv4:",conv4.shape)
        #         print("max4:",max4.shape)
        conv5 = self.c5(max4)
        max5 = self.m5(conv5)
        #         print("conv5:",conv5.shape)
        #         print("max5:",max5.shape)

        center = self.cen(max5)
        #         print("center:",center.shape)

        ###########  get gating   #############
        gating = self.gating_signal(center)
        #         print("gating:",gating.shape)

        ########### Attention Mechanism  #############
        g_conv5 = self.attentionblock5(conv5, gating)
        g_conv4 = self.attentionblock4(conv4, gating)
        g_conv3 = self.attentionblock3(conv3, gating)
        g_conv2 = self.attentionblock2(conv2, gating)

        ########## Upscaling Part (Decoder) ###########
        up5 = self.up_concat5(g_conv5, center)
        up4 = self.up_concat4(g_conv4, up5)
        up3 = self.up_concat3(g_conv3, up4)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return torch.sigmoid(final)