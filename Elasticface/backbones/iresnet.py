import torch
from torch import nn

# Define a list of symbols to be exported when using 'from backbone import *'
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100']

# Import utility functions for calculating FLOPs (Floating Point Operations)
from utils.countFLOPS import _calc_width, count_model_flops

# Utility function to create a 3x3 convolution with specified parameters
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# Utility function to create a 1x1 convolution with specified parameters
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# Squeeze-and-Excitation (SE) module for channel-wise attention
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

# Basic building block for the IResNet architecture
class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, use_se=False):
        super(IBasicBlock, self).__init__()
        # Batch normalization layer for the input
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        # First convolutional layer
        self.conv1 = conv3x3(inplanes, planes)
        # Batch normalization layer
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        # Parametric ReLU activation
        self.prelu = nn.PReLU(planes)
        # Second convolutional layer
        self.conv2 = conv3x3(planes, planes, stride)
        # Batch normalization layer
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        # Downsample layer if needed for identity mapping
        self.downsample = downsample
        # Stride for downsample layer
        self.stride = stride
        # Flag for using Squeeze-and-Excitation module
        self.use_se = use_se
        # Add SE block if specified
        if use_se:
            self.se_block = SEModule(planes, 16)

    def forward(self, x):
        # Save the input for the identity mapping
        identity = x
        # First batch normalization layer
        out = self.bn1(x)
        # First convolutional layer
        out = self.conv1(out)
        # Second batch normalization layer
        out = self.bn2(out)
        # Parametric ReLU activation
        out = self.prelu(out)
        # Second convolutional layer
        out = self.conv2(out)
        # Third batch normalization layer
        out = self.bn3(out)
        # Apply Squeeze-and-Excitation block if specified
        if self.use_se:
            out = self.se_block(out)
        # Downsample the identity if needed for identity mapping
        if self.downsample is not None:
            identity = self.downsample(x)
        # Perform element-wise addition with the identity
        out += identity
        return out

# IResNet architecture with variations in number of layers
class IResNet(nn.Module):
    # Scaling factor for fully connected layer
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, use_se=False):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.use_se = use_se
        # If specified, replace stride with dilation
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        # Layers of basic blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, use_se=self.use_se)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       use_se=self.use_se)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       use_se=self.use_se)
        # Final batch normalization layer
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        # Fully connected layer
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        # Features batch normalization
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # Utility function to create layers
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_se=False):
        downsample = None
        previous_dilation = self.dilation
        # If dilation is specified, update the dilation factor
        if dilate:
            self.dilation *= stride
            stride = 1
        # If stride is not 1 or number of input channels doesn't match, create a downsample layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        # Create layers
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation, use_se=use_se))
        return nn.Sequential(*layers)

    # Forward pass of the model
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

# Function to instantiate IResNet models with different configurations
def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()  # Placeholder, pretrained models not implemented yet
    return model

# Functions to instantiate specific variants of IResNet models
def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)

# Function for testing the models
def _test():
    import torch

    pretrained = False

    models = [
        iresnet100
    ]

    for model in models:
        net = model()
        print(net)
        # net.train()
        weight_count = _calc_width(net)
        flops = count_model_flops(net)
        print("m={}, {}".format(model.__name__, weight_count))
        print("m={}, {}".format(model.__name__, flops))
        net.eval()

        x = torch.randn(1, 3, 112, 112)

        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 512))


if __name__ == "__main__":
    _test()
