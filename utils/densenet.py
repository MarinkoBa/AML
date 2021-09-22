import torch.nn as nn
import torch

DENSENET121 = 'densenet-121'
DENSENET169 = 'densenet-169'
DENSENET201 = 'densenet-201'
DENSENET264 = 'densenet-264'


class DenseNet(nn.Module):
    '''
    Implementation of the DenseNet, which was proposed by Huang et al.
    https://arxiv.org/abs/1608.06993.
    '''

    def __init__(self, device, num_classes=3, device_ids=[0], growth_rate=32, architecture='densenet-121'):
        """
            Parameters
            ----------
            device: device
                Used device (values can be detected automatically by torch.device())
            num_classes : int
                amount of classification classes
            device_ids : array
                Contains indices of the devices
            growth_rate : int
                Amount of feature maps, which is produced by each Dense Layer. Hyperparameter of the network.
            architecture : str
                Determines the type of the DenseNet architecture.
                The different architectures has a different amount of layers.
                They were proposed by Huang et al. in https://arxiv.org/abs/1608.06993.

                Architectures:
                    - 'densenet-121'
                    - 'densenet-169'
                    - 'densenet-201'
                    - 'densenet-264'

        """
        super(DenseNet, self).__init__()

        # determine type of DenseNet architecture
        if architecture is DENSENET169:
            layers = [6, 12, 32, 32]
        elif architecture is DENSENET201:
            layers = [6, 12, 48, 32]
        elif architecture is DENSENET264:
            layers = [6, 12, 64, 48]
        else:
            # Default: DENSENET-121
            layers = [6, 12, 24, 16]

        self.device = device
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.layers = layers

        # first convolution

        # 3 input channels
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2))
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        # Dense Blocks
        self.dense_block_1 = DenseBlock(in_channels=64, n_layers=self.layers[0], growth_rate=self.growth_rate,
                                        device=device,
                                        device_ids=device_ids)

        in_channel_1 = int((64 + self.growth_rate * self.layers[0]) * 0.5)

        self.dense_block_2 = DenseBlock(in_channels=in_channel_1,
                                        n_layers=self.layers[1], growth_rate=self.growth_rate,
                                        device=device,
                                        device_ids=device_ids)

        in_channel_2 = int((in_channel_1 + self.growth_rate * self.layers[1]) * 0.5)

        self.dense_block_3 = DenseBlock(in_channels=in_channel_2, n_layers=self.layers[2], growth_rate=self.growth_rate,
                                        device=device,
                                        device_ids=device_ids)

        in_channel_3 = int((in_channel_2 + self.growth_rate * self.layers[2]) * 0.5)

        self.dense_block_4 = DenseBlock(in_channels=in_channel_3, n_layers=self.layers[3], growth_rate=self.growth_rate,
                                        device=device,
                                        device_ids=device_ids)
        channels = in_channel_1 * 2

        # Transition Layers
        self.transition1 = nn.DataParallel(TransitionLayer(in_channels=in_channel_1 * 2),
                                           device_ids=device_ids).to(device)

        self.transition2 = nn.DataParallel(TransitionLayer(in_channels=in_channel_2 * 2),
                                           device_ids=device_ids).to(device)

        self.transition3 = nn.DataParallel(TransitionLayer(in_channels=in_channel_3 * 2),
                                           device_ids=device_ids).to(device)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_channel_4 = int(in_channel_3 + self.growth_rate * self.layers[3])

        self.linear = nn.Linear(in_channel_4, self.num_classes)

        self.softmax = nn.Softmax()

    def forward(self, x):
        # initial Convolution, after this layer output should be 56x56
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.max_pool(x)

        # DenseBlock 1
        x = self.dense_block_1(x)

        # Transition 1
        x = self.transition1(x)

        # DenseBlock 2
        x = self.dense_block_2(x)

        # Transition 2
        x = self.transition2(x)

        # DenseBlock 3
        x = self.dense_block_3(x)

        # Transition 3
        x = self.transition3(x)

        # DenseBlock 4
        x = self.dense_block_4(x)

        # Classification
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        # x = self.softmax(x)

        return x


class DenseBlock(nn.Module):
    """
    A DenseBlock consists of multiple DenseLayers.
    Here the Amount of feature maps will be increased.
    """
    def __init__(self, in_channels, n_layers, growth_rate, device, device_ids=[0]):
        """
            Parameters
            ----------
            in_channels : int
                amount of incoming feature_maps
            n_layers : int
                amount of DenseLayers in this DenseBlock
            growth_rate : int
                Amount of feature maps, which is produced by each Dense Layer. Hyperparameter of the network.
            device: device
                Used device (values can be detected automatically by torch.device())
            device_ids : array
                Contains indices of the devices
        """
        super(DenseBlock, self).__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.dense_layers = nn.ModuleList()

        for i in range(self.n_layers):
            model = DenseLayer(int(i * self.growth_rate + self.in_channels), self.growth_rate)
            # model = model.double()
            dense_layer = nn.DataParallel(model, device_ids=device_ids)
            dense_layer.to(device)
            self.dense_layers.append(dense_layer)

    def forward(self, x):
        for i in range(self.n_layers):
            # in_channels are out_channels from the last layer
            y = self.dense_layers[i](x)
            x = torch.cat([x, y], dim=1)
        return x


class DenseLayer(nn.Module):
    """
    Increases the amount of incoming feature maps by growth rate.
    """
    def __init__(self, in_channels, growth_rate):
        """
            Parameters
            ----------
            in_channels : int
                amount of incoming feature_maps
            growth_rate : int
                Amount of feature maps, which is produced by each Dense Layer. Hyperparameter of the network.
        """
        super(DenseLayer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels, growth_rate, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))

        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        # first conv(1x1)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv1(x)

        # second conv(3x3)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)

        # after each Conv (3x3) dropout layer
        x = self.drop_out(x)

        return x


class TransitionLayer(nn.Module):
    """
    The Transition Layer is used to downsize the amount and size of feature maps.
    First, for downsizing the amount of Feature maps a Convolutional Layers is used.
    Second, to half the size of the feature maps an Average-Pooling_Layer is used.
    """

    def __init__(self, in_channels):
        """
            Parameters
            ----------
            in_channels : int
                amount of incoming feature_maps
        """
        super(TransitionLayer, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=(1, 1), stride=(1, 1))

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        # Convolution
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)

        # Pooling
        x = self.avg_pool(x)

        return x
