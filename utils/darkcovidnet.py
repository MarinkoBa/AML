import torch.nn as nn
import torch


class DarkCovidNet(nn.Module):
    '''
        DarkCovidNet was developed by Ozturk 2020 based on the Darknet-19 model.
        Model was created to fulfill classification task of 2D-CT images for the cases of binary (Covid, No-Finings) and multi-classification (Covid, No-Finings, Pneumonia).

        Layer layout of the network:
        C: Convolutional Layer (17-layers)
        M: Max-Pooling (5-layers)
        []: notes Blocks of Conv-layers

        C1-M1-C2-M2-[C3-C4-C5]-M3-[C6-C7-C8]-M4-[C9-C10-C11]-M5-[C12-C13-C14]-C15-C16-C17-Flatten-Linear

    '''

    def __init__(self, in_channels, num_labels,batch_size):
        super(DarkCovidNet, self).__init__()
        # 4 Max Pooling Layers of the network, all with kernel_size = 2 and stride = 2
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Operations

        self.dn_layer_1 = DN_Layer(in_channels=in_channels, out_channels=8)
        self.dn_layer_2 = DN_Layer(in_channels=8, out_channels=16)

        # 3xConv Blocks
        self.dn_block_1 = DN_Block(in_channels=16, out_channels=32)
        self.dn_block_2 = DN_Block(in_channels=32, out_channels=64)
        self.dn_block_3 = DN_Block(in_channels=64, out_channels=128)
        self.dn_block_4 = DN_Block(in_channels=128, out_channels=256)

        self.dn_layer_3 = DN_Layer(in_channels=256, out_channels=128)
        self.dn_layer_4 = DN_Layer(in_channels=128, out_channels=256)

        # last Convolution -> maps to channels = number of classes
        self.conv = nn.Conv2d(in_channels=256, out_channels=num_labels, kernel_size=(1, 1), stride=(1, 1),padding=1)
        self.bn_layer = nn.BatchNorm2d(num_labels)
        self.relu = nn.ReLU()

        # Linear layer
        self.linear = nn.Linear(in_features=batch_size*13*13*num_labels, out_features=num_labels)

    def forward(self, x):
        # first two DN-layers (DarkNetLayers)
        x = self.dn_layer_1(x)
        x = self.max_pool_1(x)
        x = self.dn_layer_2(x)
        x = self.max_pool_2(x)

        # DN-Blocks
        x = self.dn_block_1(x)
        x = self.max_pool_3(x)
        x = self.dn_block_2(x)
        x = self.max_pool_4(x)
        x = self.dn_block_3(x)
        x = self.max_pool_5(x)
        x = self.dn_block_4(x)

        # Last two DN-layers
        x = self.dn_layer_3(x)
        x = self.dn_layer_4(x)

        # last Conv-layer to reduce channels equal to num_labels
        x = self.conv(x)
        x = self.bn_layer(x)
        x = self.relu(x)

        # Flatten
        x = torch.flatten(x)

        # Linear
        x = self.linear(x)

        return x


class DN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        super(DN_Layer, self).__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.batch_normalization = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normalization(x)
        x = self.leaky_relu(x)

        return x


class DN_Block(nn.Module):
    '''
    3x DN_Layer with the same setup three times
    '''

    def __init__(self, in_channels, out_channels):
        super(DN_Block, self).__init__()
        self.dn_layer1 = DN_Layer(in_channels=in_channels, out_channels=out_channels)
        self.dn_layer2 = DN_Layer(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.dn_layer3 = DN_Layer(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.dn_layer1(x)
        x = self.dn_layer2(x)
        x = self.dn_layer3(x)
        return x
