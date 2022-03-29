import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply channel-wise gaussian smoothing on a 2d or 3d tensor.

    Adapted from:
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10

    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the Gaussian kernel.
        sigma (float, sequence): Standard deviation of the Gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 3 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The Gaussian kernel is the product of the Gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        # Padding for output to equal input dimensions
        self.padding = kernel_size[0] // 2
        if dim == 2:
            self.pad_tuple = tuple([self.padding] * 4)
            self.conv = F.conv2d
        elif dim == 3:
            self.pad_tuple = tuple([self.padding] * 6)
            self.conv = F.conv3d
        else:
            raise RuntimeError('Spatial dimensions of 2 or 3 supported. Received {}'.format(dim))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        # Use mode='replicate' to ensure image boundaries are not degraded in output for background
        # channel
        smoothed_out = self.conv(F.pad(input, self.pad_tuple, mode='replicate'),
                           weight=self.weight, groups=self.groups)

        return smoothed_out
