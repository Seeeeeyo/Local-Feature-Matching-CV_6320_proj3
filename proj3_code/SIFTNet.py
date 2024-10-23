#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from torch import nn
from proj3_code.torch_layer_utils import ImageGradientsLayer
from math import pi, floor

"""
Authors: John Lambert, Vijay Upadhya, Patsorn Sangkloy, Cusuh Ham,
Frank Dellaert, September 2019.

Implement the SIFT Deep Net that accomplishes the identical operations as the
original SIFT algorithm (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
"""


class HistogramLayer(nn.Module):
    def __init__(self) -> None:
        """
        Initialize parameter-less histogram layer, that accomplishes
        per-channel binning.
        """
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        """
        Perform a forward pass of the histogram/binning layer.
        The input should have 10 channels, where the first 8 represent cosines
        values of angles between unit circle basis vectors and image gradient
        vectors, and the last two channels will represent the (dx, dy).
        """
        cosines = x[:, :8, :, :]  # Contains gradient projections
        im_grads = x[:, 8:, :, :]  # Contains dx, dy

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        grad_magnitudes = torch.norm(im_grads, dim=1, keepdim=True)
        max_channel_indices = torch.argmax(cosines, dim=1, keepdim=True)
        binary_occupancy = torch.zeros_like(cosines).scatter_(1, max_channel_indices, 1)
        per_px_histogram = grad_magnitudes * binary_occupancy
        #######################################################################

        return per_px_histogram


class SubGridAccumulationLayer(nn.Module):
    """
    Given 8-dimensional feature vectors at each pixel, accumulate features over 4x4 subgrids.
    """

    def __init__(self) -> None:
        super().__init__()
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        self.layer = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=4,
            padding=(2, 2),
            groups=8,
            bias=False,
            stride=1,
        )
        self.layer.weight = nn.Parameter(torch.ones(8, 1, 4, 4))
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def angles_to_vectors_2d_pytorch(angles: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to 2-d basis vectors.
    Args:
    -   angles: Torch tensor of shape (N,) representing N angles
    Returns:
    -   angle_vectors: Torch tensor of shape (N,2), representing unit vectors
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    angle_vectors = torch.stack((cosines, sines), dim=1)
    ###########################################################################
    return angle_vectors


class SIFTOrientationLayer(nn.Module):
    """
    SIFT analyzes image gradients according to 8 bins, around the unit circle.
    """

    def __init__(self):
        super().__init__()
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        self.layer = nn.Conv2d(
            in_channels=2, out_channels=10, kernel_size=1, bias=False
        )
        self.layer.weight = self.get_orientation_bin_weights()
        #######################################################################

    def get_orientation_bin_weights(self) -> torch.nn.Parameter:
        """
        Populate the conv layer weights for the
        A 1x1 convolution layer with 10 orientation bins spaced by pi/8.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        angles = torch.Tensor(
            [
                np.pi / 8,
                np.pi / 4 + np.pi / 8,
                np.pi / 2 + np.pi / 8,
                3 * np.pi / 4 + np.pi / 8,
                np.pi + np.pi / 8,
                5 * np.pi / 4 + np.pi / 8,
                3 * np.pi / 2 + np.pi / 8,
                7 * np.pi / 4 + np.pi / 8,
            ]
        )
        angle_vectors = angles_to_vectors_2d_pytorch(angles)
        x, y = torch.Tensor([1, 0]).view(1, 2), torch.Tensor([0, 1]).view(1, 2)
        weight_param = torch.nn.Parameter(
            torch.cat((angle_vectors, x, y), dim=0).view(10, 2, 1, 1)
        )
        #######################################################################
        return weight_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class SIFTNet(nn.Module):
    def __init__(self):
        super().__init__()
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        self.net = nn.Sequential(
            ImageGradientsLayer(),
            SIFTOrientationLayer(),
            HistogramLayer(),
            SubGridAccumulationLayer(),
        )
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_sift_subgrid_coords(x_center: int, y_center: int):
    """
    Given the center point of a 16x16 patch, we want to pull out the
    accumulated values for each of the 16 subgrids.
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    x_for_grid = np.linspace(x_center - 6, x_center + 6, 4)
    y_for_grid = np.linspace(y_center - 6, y_center + 6, 4)
    x_grid, y_grid = np.meshgrid(x_for_grid, y_for_grid)
    x_grid = x_grid.flatten().astype(np.int64)
    y_grid = y_grid.flatten().astype(np.int64)
    ###########################################################################
    return x_grid, y_grid


def get_siftnet_features(
    img_bw: torch.Tensor, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Given a list of (x,y) coordinates, pull out the SIFT features within the
    16x16 neighborhood around each (x,y) coordinate pair.
    """
    assert img_bw.shape[0] == 1
    assert img_bw.shape[1] == 1
    assert img_bw.dtype == torch.float32
    net = SIFTNet()

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    features_all = []
    features_n = []
    features = net(img_bw)
    for i in range(x.shape[0]):
        grid_x, grid_y = get_sift_subgrid_coords(x[i], y[i])
        for j in range(grid_x.shape[0]):
            extract = features[:, :, grid_y[j], grid_x[j]]
            features_n.append(extract)
        features_all.append(
            nn.functional.normalize(torch.cat(features_n, dim=1), dim=1) ** 0.9
        )
        features_n = []

    fvs = torch.cat(features_all, dim=0).detach()

    ###########################################################################
    return fvs
