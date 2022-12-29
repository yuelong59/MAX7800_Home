###################################################################################################
# 1、指纹识别网络
###################################################################################################
# import torch
# import torch.nn as nn

# import ai8x

# class AI85Net_GARBAGE(nn.Module):
#     """
#     5-Layer CNN that uses max parameters in AI84
#     """
#     def __init__(self, num_classes=6, num_channels=3, dimensions= (3, 112, 112),
#                  planes=32, pool=2, fc_inputs=256, bias=False, **kwargs):
#         super().__init__()

#         # Limits
#         assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
#         assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

#         # Keep track of image dimensions so one constructor works for all image sizes
#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=True, **kwargs)
#         self.conv2 = ai8x.FusedConv2dReLU(16, 32, 3, stride=2, padding=1, bias=True, **kwargs)
#         self.conv3 = ai8x.FusedConv2dReLU(32, 64, 3, stride=2, padding=1, bias=True, **kwargs)
#         self.conv4 = ai8x.FusedMaxPoolConv2d(64, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=True, **kwargs)
#         # self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
#         #                                            stride=1, padding=1, bias=bias, **kwargs)

#         self.pooling=ai8x.MaxPool2d(8)
        
#         self.avg = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        
#         self.fc1 = ai8x.FusedLinearReLU(128, 128, bias=True, **kwargs)
#         self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         # print(x.size())
#         x = self.pooling(x)
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc(x)

#         return x

    
# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85Net_GARBAGE(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 1,
#     },

# ]



###################################################################################################
# 2、残差方法
###################################################################################################
"""
SimpleNet_v1 network with added residual layers for AI85.
Simplified version of the network proposed in [1].

[1] HasanPour, Seyyed Hossein, et al. "Lets keep it simple, using simple architectures to
    outperform deeper and more complex architectures." arXiv preprint arXiv:1608.06037 (2016).
"""
# from torch import nn

# import ai8x


# class AI85ResidualSimpleNet(nn.Module):
#     """
#     Residual SimpleNet v1 Model
#     """
#     def __init__(
#             self,
# #             num_classes=100,
#             num_classes=6,
#             num_channels=3,
# #             dimensions=(32, 32),  # pylint: disable=unused-argument
#             dimensions= (3, 32, 32),
#             bias=False,
#             **kwargs
#     ):
#         super().__init__()

#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias,
#                                           **kwargs)
#         self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid1 = ai8x.Add()
#         self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid2 = ai8x.Add()
#         self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid3 = ai8x.Add()
#         self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 96, 3, pool_size=2, pool_stride=2,
#                                                   stride=1, padding=1, bias=bias, **kwargs)
#         self.conv11 = ai8x.FusedMaxPoolConv2dReLU(96, 512, 1, pool_size=2, pool_stride=2,
#                                                   padding=0, bias=bias, **kwargs)
#         self.conv12 = ai8x.FusedConv2dReLU(512, 128, 1, stride=1, padding=0, bias=bias, **kwargs)
#         self.conv13 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 3, pool_size=2, pool_stride=2,
#                                                   stride=1, padding=1, bias=bias, **kwargs)
#         self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
#                                   wide=True, **kwargs)
        
#         self.avg = nn.AdaptiveAvgPool2d(1)#自适应平均池化
#         self.fc1 = ai8x.FusedLinearReLU(num_classes, 128, bias=True, **kwargs)
#         self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)          # 16x32x32
#         x_res = self.conv2(x)      # 20x32x32
#         x = self.conv3(x_res)      # 20x32x32
#         x = self.resid1(x, x_res)  # 20x32x32
#         x = self.conv4(x)          # 20x32x32
#         x_res = self.conv5(x)      # 20x16x16
#         x = self.conv6(x_res)      # 20x16x16
#         x = self.resid2(x, x_res)  # 20x16x16
#         x = self.conv7(x)          # 44x16x16
#         x_res = self.conv8(x)      # 48x8x8
#         x = self.conv9(x_res)      # 48x8x8
#         x = self.resid3(x, x_res)  # 48x8x8
#         x = self.conv10(x)         # 96x4x4
#         x = self.conv11(x)         # 512x2x2
#         x = self.conv12(x)         # 128x2x2
#         x = self.conv13(x)         # 128x1x1
#         x = self.conv14(x)         # num_classesx1x1
        
#         x = self.avg(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc(x)
#         return x


# # def ai85ressimplenet(pretrained=False, **kwargs):
# #     """
# #     Constructs a Residual SimpleNet v1 model.
# #     """
# #     assert not pretrained
# #     return AI85ResidualSimpleNet(**kwargs)


# # models = [
# #     {
# #         'name': 'ai85ressimplenet',
# #         'min_input': 1,
# #         'dim': 2,
# #     },
# # ]





# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85ResidualSimpleNet(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 2,
#     },

# ]


###################################################################################################
# 3、普通face Top1: 40.500 Params: 242352
###################################################################################################
# """
# FaceID network for AI85/AI86

# Optionally quantize/clamp activations
# """
# from torch import nn

# import ai8x


# class AI85FaceIDNet(nn.Module):
#     """
#     Simple FaceNet Model
#     """
#     def __init__(
#             self,
#             num_classes=6,  # pylint: disable=unused-argument
#             num_channels=3,
#             dimensions=(3, 220, 220),  # pylint: disable=unused-argument
#             bias=True,
#             **kwargs
#     ):
#         super().__init__()

#         # cov:(220, 220)->(220-3+2*1)/1+1=220
#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 16, 3, padding=1, bias=False, **kwargs)
#         # pool:(220, 220)->(220-2)/2+1=110
#         # cov:(110, 110)->(110-3+2*1)/1+1=110
#         self.conv2 = ai8x.FusedMaxPoolConv2dReLU(16, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=False, **kwargs)
#         # pool:(110, 110)->(110-2)/2+1=55
#         # cov:(55, 55)->(55-3+2*1)/1+1=55
#         self.conv3 = ai8x.FusedMaxPoolConv2dReLU(32, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
#         # pool:(55, 55)->(55-2)/2+1=28
#         # cov:(28, 28)->(28-3+2*1)/1+1=28
#         self.conv4 = ai8x.FusedMaxPoolConv2dReLU(32, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
#         # pool:(28, 28)->(28-2)/2+1=14
#         # cov:(14, 14)->(14-3+2*1)/1+1=14
#         self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
#         self.conv6 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
#         self.conv7 = ai8x.FusedConv2dReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        
#         # pool:(14, 14)->(14-2)/2+1=7
#         # cov:(7, 7)->(7-3+2*1)/1+1=7
#         self.conv8 = ai8x.FusedMaxPoolConv2d(64, 512, 1, pool_size=2, pool_stride=2,
#                                              padding=0, bias=False, **kwargs)
        
#         self.avg = nn.AdaptiveAvgPool2d(1)#自适应平均池化
#         self.fc1 = ai8x.FusedLinearReLU(512, 128, bias=True, **kwargs)
#         self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x) 
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
        
#         x = self.avg(x)
        
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc(x)
#         return x


# # def ai85faceidnet(pretrained=False, **kwargs):
# #     """
# #     Constructs a FaceIDNet model.
# #     """
# #     assert not pretrained
# #     return AI85FaceIDNet(**kwargs)


# # models = [
# #     {
# #         'name': 'ai85faceidnet',
# #         'min_input': 1,
# #         'dim': 3,
# #     },
# # ]

# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85FaceIDNet(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 1,
#     },

# ]


###################################################################################################
# 4、基于face的自建网络v1
###################################################################################################
# """
# FaceID network for AI85/AI86

# Optionally quantize/clamp activations
# """
# from torch import nn

# import ai8x


# class AI85FaceIDNet(nn.Module):
#     """
#     Simple FaceNet Model
#     """
#     def __init__(
#             self,
#             num_classes=6,  # pylint: disable=unused-argument
#             num_channels=3,
#             dimensions=(3, 220, 220),  # pylint: disable=unused-argument
#             bias=True,
#             **kwargs
#     ):
#         super().__init__()

#         # cov:(220, 220)->(220-3+2*1)/1+1=220
#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 96, 3, padding=1, bias=False, **kwargs)
#         # pool:(220, 220)->(220-2)/2+1=110
#         # cov:(110, 110)->(110-3+2*1)/1+1=110
#         self.conv2 = ai8x.FusedMaxPoolConv2dReLU(96, 96, 3, pool_size=2, pool_stride=2, padding=1, bias=False, **kwargs)
        
#         # cov:(110, 110)
#         self.conv3 = ai8x.FusedConv2dReLU(96, 256, 3, padding=1, bias=False, **kwargs)
#         # pool:(110, 110)->(110-2)/2+1=55
#         # cov:(55, 55)->(55-3+2*1)/1+1=55
#         self.conv4 = ai8x.FusedMaxPoolConv2dReLU(256, 256, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
#         # cov:(55, 55)
#         self.conv5 = ai8x.FusedConv2dReLU(256, 384, 3, padding=1, bias=False, **kwargs)
#         # pool:(55, 55)->(55-2)/2+1=28
#         # cov:(28, 28)->(28-3+2*1)/1+1=28
#         self.conv6 = ai8x.FusedMaxPoolConv2dReLU(384, 384, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
#         # cov:(28, 28)
#         self.conv7 = ai8x.FusedConv2dReLU(384, 1024, 3, padding=1, bias=False, **kwargs)
#         # pool:(28, 28)->(28-2)/2+1=14
#         # cov:(14, 14)->(14-3+2*1)/1+1=14        
#         self.conv8 = ai8x.FusedMaxPoolConv2dReLU(1024, 1024, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)

#         self.conv9 = ai8x.FusedConv2dReLU(1024, 1024, 3, padding=1, bias=bias, **kwargs)
#         self.conv10 = ai8x.FusedConv2dReLU(1024, 1024, 3, padding=1, bias=bias, **kwargs)
#         # pool:(14, 14)->(14-2)/2+1=7
#         # cov:(7, 7)->(7-3+2*1)/1+1=7        
#         self.conv11 = ai8x.FusedMaxPoolConv2d(1024, 1024, 1, pool_size=2, pool_stride=2, padding=0, bias=False, **kwargs)
        
#         # pool:(7, 7)->(7-2)/2+1=3
#         # cov:(3, 3)->(3-3+2*1)/1+1=3        
#         self.conv12 = ai8x.FusedMaxPoolConv2d(1024, 1024, 1, pool_size=2, pool_stride=2, padding=0, bias=False, **kwargs)
        
        
#         #自适应平均池化
#         #（1，1）
#         self.avg = nn.AdaptiveAvgPool2d(1)
        
#         # 线性层
#         self.fc1 = ai8x.FusedLinearReLU(1024, 128, bias=True, **kwargs)
#         self.fc2 = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
#         x = self.conv9(x)
#         x = self.conv10(x)
#         x = self.conv11(x)
#         x = self.conv12(x)
        
#         x = self.avg(x)
        
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
        
#         return x

    
# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85FaceIDNet(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 1,
#     },

# ]


###################################################################################################
# 5、基于AlexNEt
###################################################################################################
# """
# FaceID network for AI85/AI86

# Optionally quantize/clamp activations
# """
# from torch import nn

# import ai8x


# class AI85FaceIDNet(nn.Module):
#     """
#     Simple FaceNet Model
#     """
#     def __init__(
#             self,
#             num_classes=6,  # pylint: disable=unused-argument
#             num_channels=3,
#             dimensions=(3, 220, 220),  # pylint: disable=unused-argument
#             bias=True,
#             **kwargs
#     ):
#         super().__init__()
        
#         #大卷积核、较大的步长、较多的通道
#         # cov:(220, 220)->(220-3)/2+1=109
#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 96, 3, stride=2, **kwargs)
#         # pool:(109, 109)->(109-3)/2+1=54
#         self.pool1 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         #卷积核、步长恢复正常大小，进一步扩大通道
#         # cov:(54, 54)->(54-3+2*1))/1+1=54
#         self.conv2 = ai8x.FusedConv2dReLU(96, 256, 3, padding=1, **kwargs)
#         # pool:(54, 54)->(54-3)/2+1=26
#         self.pool2 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         #连续的卷积层，疯狂提取特征
#         # cov:(26, 26)->(26-3+2*1)/1+1=26
#         self.conv3 = ai8x.FusedConv2dReLU(256, 384, 3, padding=1, **kwargs)
#         # cov:(26, 26)->(26-3+2*1)/1+1=26
#         self.conv4 = ai8x.FusedConv2dReLU(384, 384, 3, padding=1, **kwargs)
#         # cov:(26, 26)->(26-3+2*1)/1+1=26
#         self.conv5 = ai8x.FusedConv2dReLU(384, 512, 3, padding=1, **kwargs)
#         # pool:(26, 26)->(26-3)/2+1=12
#         self.pool3 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         # cov:(12, 12)->(12-3+2*1)/1+1=12
#         self.conv6 = ai8x.FusedConv2dReLU(512, 1024, 3, padding=1, **kwargs)
#         # pool:(12, 12)->(12-3)/2+1=5
#         self.pool4 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         # cov:(5, 5)->(5-3+2*1)/1+1=5
#         self.conv7 = ai8x.FusedConv2dReLU(1024, 512, 3, padding=1, **kwargs)
#         # pool:(5, 5)->(5-3)/2+1=2
#         self.pool5 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         #自适应平均池化
#         #（1，1）
#         self.avg = nn.AdaptiveAvgPool2d(1)
        
#         #全连接层
#         self.fc1 = ai8x.Linear(128*2*2, 1024, **kwargs) #这里的上层输入是图像中的全部像素
#         self.fc2 = ai8x.Linear(1024, 1024, **kwargs)
#         self.fc3 = ai8x.Linear(1024, num_classes, **kwargs) #输出ImageNet的num_classes个类别


#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.pool3(x)
#         x = self.conv6(x)
#         x = self.pool4(x)
#         x = self.conv7(x)
#         x = self.pool5(x)
        
#         x = self.avg(x)
        
# #         x = x.view(x.size(0), -1)
#         x = x.view(-1, 128*2*2)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
        
        
#         return x

    
# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85FaceIDNet(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 1,
#     },

# ]



###################################################################################################
# 6、残差方法 V2
###################################################################################################

# from torch import nn

# import ai8x


# class AI85FaceIDNet(nn.Module):
#     """
#     Simple FaceNet Model
#     """
#     def __init__(
#             self,
#             num_classes=6,  # pylint: disable=unused-argument
#             num_channels=3,
#             dimensions=(3, 220, 220),  # pylint: disable=unused-argument
#             bias=True,
#             **kwargs
#     ):
#         super().__init__()
        
#         #先快速减小图像尺寸，降低到（26，26）
#         #大卷积核、较大的步长、较多的通道
#         # cov:(220, 220)->(220-3)/2+1=109
#         self.conv1 = ai8x.FusedConv2dReLU(num_channels, 96, 3, stride=2, **kwargs)
#         # cov:(109, 109)->(109-3)/2+1=54
#         self.conv2 = ai8x.FusedConv2dReLU(96, 128, 3, stride=2, **kwargs)
#         # pool:(54, 54)->(54-3)/2+1=26
#         self.pool1 = ai8x.MaxPool2d(kernel_size=3, stride=2, **kwargs)
        
#         # 包含残差操作
#         # cov:(26, 26)->(26-3+2*1)/1+1=26
#         self.conv3 = ai8x.FusedConv2dReLU(128, 128, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv4 = ai8x.FusedConv2dReLU(128, 128, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.conv5 = ai8x.FusedConv2dReLU(128, 128, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid1 = ai8x.Add()
        
#         # pool:(26, 26)->(26-2)/2+1=13
#         # cov:(13, 13)->(13-3+2*1)/1+1=13
#         self.conv6 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv7 = ai8x.FusedConv2dReLU(128, 128, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid2 = ai8x.Add()
        
#         # cov:(13, 13)->(13-3+2*1)/1+1=13
#         self.conv8 = ai8x.FusedConv2dReLU(128, 144, 3, stride=1, padding=1, bias=bias, **kwargs)
#         # pool:(13, 13)->(13-2)/2+1=6
#         # cov:(6, 6)->(6-3+2*1)/1+1=6
#         self.conv9 = ai8x.FusedMaxPoolConv2dReLU(144, 148, 3, pool_size=2, pool_stride=2,
#                                                  stride=1, padding=1, bias=bias, **kwargs)
#         self.conv10 = ai8x.FusedConv2dReLU(148, 148, 3, stride=1, padding=1, bias=bias, **kwargs)
#         self.resid3 = ai8x.Add()
        
#         # pool:(6, 6)->(6-2)/2+1=3
#         # cov:(3, 3)->(3-3+2*1)/1+1=3
#         self.conv11 = ai8x.FusedMaxPoolConv2dReLU(148, 196, 3, pool_size=2, pool_stride=2,
#                                                   stride=1, padding=1, bias=bias, **kwargs)
#         # pool:(3, 3)->(3-2)/2+1=1
#         # cov:(1, 1)->(1-3+2*1)/1+1=1
#         self.conv12 = ai8x.FusedMaxPoolConv2dReLU(196, 512, 1, pool_size=2, pool_stride=2,
#                                                   padding=0, bias=bias, **kwargs)
#         self.conv13 = ai8x.FusedConv2dReLU(512, 128, 1, stride=1, padding=0, bias=bias, **kwargs)
#         # cov:(1, 1)->(1-3+2*1)/1+1=1
# #         self.conv14 = ai8x.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=bias,
# #                                   wide=True, **kwargs)
        
#         #自适应平均池化
#         #（1，1）
#         self.avg = nn.AdaptiveAvgPool2d(1)
        
# #         #全连接层
# #         self.fc1 = ai8x.Linear(128, 1024, **kwargs) #这里的上层输入是图像中的全部像素
# #         self.fc2 = ai8x.Linear(1024, 1024, **kwargs)
#         self.fc3 = ai8x.Linear(128, num_classes, **kwargs) #输出ImageNet的num_classes个类别


#     def forward(self, x):  # pylint: disable=arguments-differ
#         """Forward prop"""
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.pool1(x)
#         x = self.conv3(x) 
#         x = self.conv4(x)
#         x = self.conv5(x) 
#         x = self.resid1(x)
        
#         x = self.conv6(x)
#         x = self.conv7(x) 
#         x = self.resid2(x)
        
#         x = self.conv8(x) 
#         x = self.conv9(x)
#         x = self.conv10(x) 
#         x = self.resid3(x)        

#         x = self.conv11(x)
#         x = self.conv12(x)
#         x = self.conv13(x)
# #         x = self.conv14(x)

        
#         x = self.avg(x)
        
#         x = x.view(x.size(0), -1)
# #         x = x.view(-1, 128*2*2)
# #         x = self.fc1(x)
# #         x = self.fc2(x)
#         x = self.fc3(x)
        
        
#         return x

    
# def ai85net_garbage(pretrained=False, **kwargs):
#     """
#     Constructs a AI85Net5 model.
#     """
#     assert not pretrained
#     return AI85FaceIDNet(**kwargs)

# models = [
#     {
#         'name': 'ai85net_garbage',
#         'min_input': 1,
#         'dim': 1,
#     },

# ]




###################################################################################################
# 7、普通face V2 Params: 578220
###################################################################################################
# """
# FaceID network for AI85/AI86

# Optionally quantize/clamp activations
# """
from torch import nn

import ai8x


class AI85FaceIDNet(nn.Module):
    """
    Simple FaceNet Model
    """
    def __init__(
            self,
            num_classes=6,  # pylint: disable=unused-argument
            num_channels=3,
            dimensions=(3, 220, 220),  # pylint: disable=unused-argument
            bias=True,
            **kwargs
    ):
        super().__init__()

        # cov:(220, 220)->(220-3+2*1)/1+1=220
        self.conv1 = ai8x.FusedConv2dReLU(num_channels, 64, 3, padding=1, bias=False, **kwargs)
        # pool:(220, 220)->(220-2)/2+1=110
        # cov:(110, 110)->(110-3+2*1)/1+1=110
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=False, **kwargs)
        # pool:(110, 110)->(110-2)/2+1=55
        # cov:(55, 55)->(55-3+2*1)/1+1=55
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
        # pool:(55, 55)->(55-2)/2+1=28
        # cov:(28, 28)->(28-3+2*1)/1+1=28
        self.conv4 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
        # pool:(28, 28)->(28-2)/2+1=14
        # cov:(14, 14)->(14-3+2*1)/1+1=14
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        
        # pool:(14, 14)->(14-2)/2+1=7
        # cov:(7, 7)->(7-3+2*1)/1+1=7
        # 增加网络后更改512为128
        self.conv6 = ai8x.FusedMaxPoolConv2d(64, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=False, **kwargs)
        
        # 瓶颈结构
        #cov:(7, 7)->(7-1+2*0)/1+1=7
        # cov:(7, 7)->(7-3+2*1)/1+1=7
        #cov:(7, 7)->(7-1+2*0)/1+1=7
#         self.conv7 = ai8x.FusedConv2dReLU(128, 128, 1, padding=0, bias=bias, **kwargs)
#         self.conv8 = ai8x.FusedConv2dReLU(128, 128, 3, padding=1, bias=bias, **kwargs)
#         self.conv9 = ai8x.FusedConv2dReLU(128, 128, 1, padding=0, bias=bias, **kwargs)
        
        # pool:(7, 7)->(7-2)/2+1=3
        # cov:(3, 3)->(3-1+2*0)/1+1=3
        self.conv10 = ai8x.FusedMaxPoolConv2d(128, 256, 1, pool_size=2, pool_stride=2, padding=0, bias=False, **kwargs)
        
        self.avg = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        self.fc1 = ai8x.FusedLinearReLU(256, 128, bias=True, **kwargs)
        self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
#         x = self.conv9(x)
        x = self.conv10(x)
#         x = self.conv11(x)
        
        x = self.avg(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc(x)
        return x


def ai85net_garbage(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85FaceIDNet(**kwargs)

models = [
    {
        'name': 'ai85net_garbage',
        'min_input': 1,
        'dim': 1,
    },

]
