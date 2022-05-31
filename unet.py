from unet_blocks import *
import torch.nn.functional as F
from bayesian import GaussianConv2d, GaussianLinear

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padding: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        # Downsampling path
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool, vb=False))

        # Upsampling path
        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            upblock = None
            if i == 0:
                # only apply VB to last layer
                upblock = UpConvBlock(input, output, initializers, padding, vb=True)
            else:
                upblock = UpConvBlock(input, output, initializers, padding, vb=False)
            self.upsampling_path.append(upblock)

        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)

    
    # gaussian prior
    def svd_regularizer(self, mu, log_sigma):
        logvar = 2 * log_sigma
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def gaussian_layers(self):
        for module in self.modules():
            if type(module) == GaussianConv2d or type(module) == GaussianLinear: 
                yield module

    # Return sum of kls from all variational dropout layers
    def regularizer(self):
        kl = 0.0
        for module in self.gaussian_layers():
            kl += self.svd_regularizer(module.weight, module.log_sigma)
        return kl

    # Enable sampling of weights in U-Net
    def _sample_on_eval(self, enabled):
        for module in self.gaussian_layers():
            module.sample_on_eval = enabled

    # Set deterministic of weights in U-Net
    def _set_deterministic(self, is_deterministic=True):
        for module in self.gaussian_layers():
            module.deterministic = is_deterministic

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        # Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x

