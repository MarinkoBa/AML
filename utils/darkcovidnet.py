import torch.nn as nn


class DarkCovidNet(nn.Module):
    '''
        DarkCovidNet was developed by Ozturk 2020 based on the Darknet-19 model.
        Model was created to fulfill classification task of 2D-CT images for the cases of binary (Covid, No-Finings) and multi-classification (Covid, No-Finings, Pneumonia).

        Layer layout of the network:
        C: Convolutional Layer
        M: Max-Pooling

        C1-M1-C2-M2-C3-C4-C5-M3-C6-C7-C8-M4-C9-C10-C11-M5-C12-C13-C14-C15-C16-C17-Flatten-Linear

        '''

    def __init__(self):
        super(DarkCovidNet, self).__init__()
        # 4 Max Pooling Layers of the network
        self.max_pool_1 = nn.MaxPool2d()
        self.max_pool_2 = nn.MaxPool2d()
        self.max_pool_3 = nn.MaxPool2d()
        self.max_pool_4 = nn.MaxPool2d()

        self.DN_layer_1 = DN_Layer()

    def forward(self, x):
        # TODO implement forward method

        return ...


class DN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DN_Layer, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.batch_normalization = nn.BatchNorm2d()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normalization(x)
        x = self.leaky_relu(x)

        return x
