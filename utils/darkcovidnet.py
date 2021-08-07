import torch.nn as nn


class DarkCovidNet(nn.Module):


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
