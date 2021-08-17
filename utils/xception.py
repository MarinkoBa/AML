import torch
import torch.nn as nn

class XCeption(nn.Module):

    def __init__(self, device):
        super(XCeption, self).__init__()

        self.device = device

        # Entry flow pre
        self.conv2DBatchNorm1 = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(2,2)).to(device))
        self.relu1 = torch.nn.DataParallel(nn.ReLU().to(device))

        self.conv2DBatchNorm2 = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=32, out_channels=64, kernel_size=(3,3)).to(device))
        self.relu2 = torch.nn.DataParallel(nn.ReLU().to(device))

        # Entry flow 1/3
        self.residual11 = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(2,2)).to(device))


        self.separableConv11 = torch.nn.DataParallel(SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3,3)).to(device))

        self.relu11 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConv12 = torch.nn.DataParallel(SeparableConv2d(in_channels=128, out_channels=128, kernel_size=(3,3)).to(device))

        self.maxPool11 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1).to(device))

        # Entry flow 2/3
        self.residual21 = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=128, out_channels=256, kernel_size=(1,1), stride=(2,2)).to(device))


        self.relu21 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConv21 = torch.nn.DataParallel(SeparableConv2d(in_channels=128, out_channels=256, kernel_size=(3,3)).to(device))

        self.relu22 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConv22 = torch.nn.DataParallel(SeparableConv2d(in_channels=256, out_channels=256, kernel_size=(3,3)).to(device))

        self.maxPool21 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1).to(device))

        # Entry flow 3/3
        self.residual31 = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=256, out_channels=728, kernel_size=(1,1), stride=(2,2)).to(device))

        self.relu31 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConv31 = torch.nn.DataParallel(SeparableConv2d(in_channels=256, out_channels=728, kernel_size=(3,3)).to(device))

        self.relu32 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConv32 = torch.nn.DataParallel(SeparableConv2d(in_channels=728, out_channels=728, kernel_size=(3,3)).to(device))

        self.maxPool31 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1).to(device))


        # Middle flow

        self.middleFlow1 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow2 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow3 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow4 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow5 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow6 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow7 = torch.nn.DataParallel(MittleFlow().to(device))
        self.middleFlow8 = torch.nn.DataParallel(MittleFlow().to(device))


        # Exit flow
        self.residualExitFlow = torch.nn.DataParallel(Conv2DBatchNorm(in_channels=728, out_channels=1024, kernel_size=(1,1), stride=(2,2)).to(device))


        self.reluExitFlow1 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConvExitFlow1 = torch.nn.DataParallel(SeparableConv2d(in_channels=728, out_channels=728, kernel_size=(3,3)).to(device))

        self.reluExitFlow2 = torch.nn.DataParallel(nn.ReLU().to(device))
        self.separableConvExitFlow2 = torch.nn.DataParallel(SeparableConv2d(in_channels=728, out_channels=1024, kernel_size=(3,3)).to(device))

        self.maxPoolExitFlow = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1).to(device))

        # Exit flow post
        self.separableConvExitFlowPost1 = torch.nn.DataParallel(SeparableConv2d(in_channels=1024, out_channels=1536, kernel_size=(3,3)).to(device))
        self.reluExitFlowPost1 = torch.nn.DataParallel(nn.ReLU().to(device))

        self.separableConvExitFlowPost2 = torch.nn.DataParallel(SeparableConv2d(in_channels=1536, out_channels=2048, kernel_size=(3,3)).to(device))
        self.reluExitFlowPost2 = torch.nn.DataParallel(nn.ReLU().to(device))

        self.globalAvgPool = torch.nn.DataParallel(nn.AvgPool2d(kernel_size=(10,10), padding=0).to(device))

        self.denseLayer1 = torch.nn.DataParallel(nn.Linear(in_features=2048, out_features=256).to(device))

        self.leakyRelu = torch.nn.DataParallel(nn.LeakyReLU().to(device))

        self.denseLayer2 = torch.nn.DataParallel(nn.Linear(in_features=256, out_features=3).to(device))

        self.softmax = torch.nn.DataParallel(nn.Softmax().to(device))


    def forward(self, x, batch_size):

        # Entry flow pre
        x = self.conv2DBatchNorm1(x)
        x = self.relu1(x)

        x = self.conv2DBatchNorm2(x)
        x = self.relu2(x)

        # Entry flow 1/3
        residual = self.residual11(x)


        x = self.separableConv11(x)

        x = self.relu11(x)
        x = self.separableConv12(x)

        x = self.maxPool11(x)

        x = torch.add(x, residual)

        # Entry flow 2/3
        residual = self.residual21(x)


        x = self.relu21(x)
        x = self.separableConv21(x)

        x = self.relu22(x)
        x = self.separableConv22(x)

        x = self.maxPool21(x)

        x = torch.add(x, residual)

        # Entry flow 3/3
        residual = self.residual31(x)

        x = self.relu31(x)
        x = self.separableConv31(x)

        x = self.relu32(x)
        x = self.separableConv32(x)

        x = self.maxPool31(x)

        residual = torch.add(x, residual)

        # Middle flow

        x = self.middleFlow1(residual)

        residual = torch.add(x, residual)
        x = self.middleFlow2(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow3(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow4(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow5(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow6(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow7(residual)
        residual = torch.add(x, residual)

        x = self.middleFlow8(residual)
        x = torch.add(x, residual)


        # Exit flow
        residual = self.residualExitFlow(x)

        x = self.reluExitFlow1(x)
        x = self.separableConvExitFlow1(x)

        x = self.reluExitFlow2(x)
        x = self.separableConvExitFlow2(x)

        x = self.maxPoolExitFlow(x)

        x = torch.add(x, residual)


        # Exit flow post

        x = self.separableConvExitFlowPost1(x)
        x = self.reluExitFlowPost1(x)

        x = self.separableConvExitFlowPost2(x)
        x = self.reluExitFlowPost2(x)

        x = self.globalAvgPool(x)

        x = x.view(batch_size, 2048)

        x = self.denseLayer1(x)

        x = self.leakyRelu(x)

        x = self.denseLayer2(x)

        #x = self.softmax(x)
        return x


class MittleFlow(nn.Module):

    def __init__(self):
        super(MittleFlow, self).__init__()

        self.relu1 = nn.ReLU()
        self.separableConv1 = SeparableConv2d(in_channels=728, out_channels=728, kernel_size=(3,3))

        self.relu2 = nn.ReLU()
        self.separableConv2 = SeparableConv2d(in_channels=728, out_channels=728, kernel_size=(3,3))

        self.relu3 = nn.ReLU()
        self.separableConv3 = SeparableConv2d(in_channels=728, out_channels=728, kernel_size=(3,3))


    def forward(self, x):

        x = self.relu1(x)
        x = self.separableConv1(x)

        x = self.relu2(x)
        x = self.separableConv2(x)

        x = self.relu3(x)
        x = self.separableConv3(x)

        return x


class Conv2DBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0):
        super(Conv2DBatchNorm, self).__init__()

        self.conv2D = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchNorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2D(x)
        x = self.batchNorm(x)
        return x

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x