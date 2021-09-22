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

    def __init__(self, in_channels, num_labels, device):
        """
            Parameters
            ----------
            in_channels : int
                Amount of incoming feature_maps.
            num_labels : int
                Number of classification classes.
            device: str
                'cuda' for gpu or 'cpu' for CPU (values can be detected automatically by torch.device())
        """
        super(DarkCovidNet, self).__init__()

        self.device = device

        # 4 Max Pooling Layers of the network, all with kernel_size = 2 and stride = 2
        self.max_pool_1 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=2, stride=2).to(device))
        self.max_pool_2 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=2, stride=2).to(device))
        self.max_pool_3 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=2, stride=2).to(device))
        self.max_pool_4 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=2, stride=2).to(device))
        self.max_pool_5 = torch.nn.DataParallel(nn.MaxPool2d(kernel_size=2, stride=2).to(device))

        # Convolutional Operations

        self.dn_layer_1 = torch.nn.DataParallel(DN_Layer(in_channels=in_channels, out_channels=8).to(device))
        self.dn_layer_2 = torch.nn.DataParallel(DN_Layer(in_channels=8, out_channels=16).to(device))

        # 3xConv Blocks
        self.dn_block_1 = torch.nn.DataParallel(DN_Block(in_channels=16, out_channels=32).to(device))
        self.dn_block_2 = torch.nn.DataParallel(DN_Block(in_channels=32, out_channels=64).to(device))
        self.dn_block_3 = torch.nn.DataParallel(DN_Block(in_channels=64, out_channels=128).to(device))
        self.dn_block_4 = torch.nn.DataParallel(DN_Block(in_channels=128, out_channels=256).to(device))

        self.dn_layer_3 = torch.nn.DataParallel(DN_Layer(in_channels=256, out_channels=128).to(device))
        self.dn_layer_4 = torch.nn.DataParallel(DN_Layer(in_channels=128, out_channels=256).to(device))

        # last Convolution -> maps to channels = number of classes
        self.conv = torch.nn.DataParallel(nn.Conv2d(in_channels=256, out_channels=num_labels, kernel_size=(1, 1), stride=(1, 1),padding=1).to(device))
        self.bn_layer = torch.nn.DataParallel(nn.BatchNorm2d(num_labels).to(device))
        self.relu = torch.nn.DataParallel(nn.ReLU().to(device))

        # Linear layer
        self.linear = torch.nn.DataParallel(nn.Linear(in_features=13*13*num_labels, out_features=num_labels).to(device))

        self.softmax = torch.nn.DataParallel(nn.Softmax().to(device))


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
        x = torch.flatten(x, start_dim=1)

        # Linear
        x = self.linear(x)

        #x = self.softmax(x)

        return x


class DN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1)):
        """
            Parameters
            ----------
            in_channels : int
                Amount of incoming feature_maps.
            out_channels : int
                Amount of outgoing feature_maps.
            kernel_size : Tuple (int,int)
                Kernelsize of the 2D-Conv.-Layer.
            stride : Tuple (int,int)
                Size of the kernel stride of the Conv.-Layer
        """
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
        """
            Parameters
            ----------
            in_channels : int
                Amount of incoming feature_maps.
            out_channels : int
                Amount of outgoing feature_maps.
        """

        super(DN_Block, self).__init__()
        self.dn_layer1 = DN_Layer(in_channels=in_channels, out_channels=out_channels)
        self.dn_layer2 = DN_Layer(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.dn_layer3 = DN_Layer(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.dn_layer1(x)
        x = self.dn_layer2(x)
        x = self.dn_layer3(x)
        return x
