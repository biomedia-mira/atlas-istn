import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineSTN2D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(AffineSTN2D, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

        # Encoder
        use_bias = True
        self.down1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 2)
        self.rotation = nn.Linear(32, 1)
        self.scaling = nn.Linear(32, 2)
        # self.shearing = nn.Linear(32, 1)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=self.dtype))
        # self.shearing.weight.data.zero_()
        # self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))

        # predict affine parameters
        xa = F.avg_pool2d(x4, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        b, c, h, w = x.size()
        id_grid = self.grid.unsqueeze(0).repeat(b, 1, 1, 1)
        id_grid = id_grid.view(b, 2, -1)

        ones  = torch.ones([b, 1, id_grid.size(2)]).to(self.device)

        self.T = torch.bmm(self.theta[:, 0:2, :], torch.cat((id_grid, ones), dim=1))
        self.T = self.T.view(b, 2, h, w)
        self.T = self.move_grid_dims(self.T)

        self.T_inv = torch.bmm(self.theta_inv[:, 0:2, :], torch.cat((id_grid, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 2, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def regularizer(self):
        return torch.tensor([0], dtype=self.dtype).to(self.device)

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        # shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        # shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
        # shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
        # shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
        # shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
        # shearing_matrix[:, 2, 2] = 1.0

        # Affine transform
        # matrix = torch.bmm(shearing_matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Linear transform - no shearing
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class DiffeomorphicSTN2D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(DiffeomorphicSTN2D, self).__init__()

        self.device = device
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        # Encoder-decoder
        dim = 2
        use_bias = True
        self.down1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv23 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv2d(8, dim, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='bilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='bilinear', align_corners=False)

        self.T = self.move_grid_dims(self.grid + self.disp)
        self.T_inv = self.move_grid_dims(self.grid + self.disp_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :] - self.disp[:, :, :-1, :])
        dx = torch.abs(self.disp[:, :, :, 1:] - self.disp[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class FullSTN2D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(FullSTN2D, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

        # Encoder-decoder
        dim = 2
        use_bias = True
        self.down1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv32 = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv23 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv2d(8, dim, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 2)
        self.rotation = nn.Linear(32, 1)
        self.scaling = nn.Linear(32, 2)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=self.dtype))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # predict affine parameters
        xa = F.avg_pool2d(x4, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='bilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='bilinear', align_corners=False)

        # compose transformations
        b, c, h, w = x.size()
        warp_field = self.grid + self.disp
        warp_field_inv = self.grid + self.disp_inv

        ones = torch.ones([b, 1, warp_field.view(b, 2, -1).size(2)]).to(self.device)

        # compose forward transformation
        grid_affine = self.grid.unsqueeze(0).repeat(b, 1, 1, 1)
        grid_affine = torch.bmm(self.theta[:, 0:2, :], torch.cat((grid_affine.view(b, 2, -1), ones), dim=1))
        self.T = self.transform(warp_field, self.move_grid_dims(grid_affine.view(b, 2, h, w)))
        self.T = self.move_grid_dims(self.T)

        # compose backward transformation
        warp_field_inv = warp_field_inv.view(b, 2, -1)
        self.T_inv = torch.bmm(self.theta_inv[:, 0:2, :], torch.cat((warp_field_inv, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 2, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :] - self.disp[:, :, :-1, :])
        dx = torch.abs(self.disp[:, :, :, 1:] - self.disp[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 2] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
        rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
        rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
        rotation_matrix[:, 2, 2] = 1.0

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = 1.0

        # Affine transform
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class AffineSTN3D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(AffineSTN3D, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

        # Encoder
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 3)
        self.rotation = nn.Linear(32, 3)
        self.scaling = nn.Linear(32, 3)
        # self.shearing = nn.Linear(32, 3).to(self.device)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        # self.shearing.weight.data.zero_()
        # self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))

        # predict affine parameters
        xa = F.avg_pool3d(x4, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        b, c, d, h, w = x.size()
        id_grid = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        id_grid = id_grid.view(b, 3, -1)

        ones  = torch.ones([b, 1, id_grid.size(2)]).to(self.device)

        self.T = torch.bmm(self.theta[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
        self.T = self.T.view(b, 3, d, h, w)
        self.T = self.move_grid_dims(self.T)

        self.T_inv = torch.bmm(self.theta_inv[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 3, d, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def regularizer(self):
        return torch.tensor([0], dtype=self.dtype).to(self.device)

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        #
        # shear_1 = shear[:, 0].view(-1)
        # shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        # shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        # shearing_matrix_1[:, 0, 0] = 1.0
        # shearing_matrix_1[:, 3, 3] = 1.0
        #
        # shear_2 = shear[:, 1].view(-1)
        # shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        # shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        # shearing_matrix_2[:, 1, 1] = 1.0
        # shearing_matrix_2[:, 3, 3] = 1.0
        #
        # shear_3 = shear[:, 2].view(-1)
        # shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        # shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        # shearing_matrix_3[:, 2, 2] = 1.0
        # shearing_matrix_3[:, 3, 3] = 1.0
        #
        # shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        # shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        # matrix = torch.bmm(shearing_matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Linear transform - no shearing
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class AffineSTN3Dv2(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(AffineSTN3Dv2, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

        # Encoder
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down4 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 3)
        self.rotation = nn.Linear(32, 3)
        self.scaling = nn.Linear(32, 3)
        # self.shearing = nn.Linear(32, 3).to(self.device)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        # self.shearing.weight.data.zero_()
        # self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x5 = self.down4(x4)
        x5 = F.relu(self.conv51(x5))

        # predict affine parameters
        xa = F.avg_pool3d(x5, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        b, c, d, h, w = x.size()
        id_grid = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        id_grid = id_grid.view(b, 3, -1)

        ones  = torch.ones([b, 1, id_grid.size(2)]).to(self.device)

        self.T = torch.bmm(self.theta[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
        self.T = self.T.view(b, 3, d, h, w)
        self.T = self.move_grid_dims(self.T)

        self.T_inv = torch.bmm(self.theta_inv[:, 0:3, :], torch.cat((id_grid, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 3, d, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def regularizer(self):
        return torch.tensor([0], dtype=self.dtype).to(self.device)

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
        #
        # shear_1 = shear[:, 0].view(-1)
        # shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        # shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        # shearing_matrix_1[:, 0, 0] = 1.0
        # shearing_matrix_1[:, 3, 3] = 1.0
        #
        # shear_2 = shear[:, 1].view(-1)
        # shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        # shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        # shearing_matrix_2[:, 1, 1] = 1.0
        # shearing_matrix_2[:, 3, 3] = 1.0
        #
        # shear_3 = shear[:, 2].view(-1)
        # shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        # shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        # shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        # shearing_matrix_3[:, 2, 2] = 1.0
        # shearing_matrix_3[:, 3, 3] = 1.0
        #
        # shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        # shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        # matrix = torch.bmm(shearing_matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, translation_matrix)

        # Linear transform - no shearing
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class DiffeomorphicSTN3D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(DiffeomorphicSTN3D, self).__init__()

        self.device = device
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        # Encoder-decoder
        dim = 3
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv23 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv3d(8, dim, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='trilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='trilinear', align_corners=False)

        self.T = self.move_grid_dims(self.grid + self.disp)
        self.T_inv = self.move_grid_dims(self.grid + self.disp_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :, :] - self.disp[:, :, :-1, :, :])
        dx = torch.abs(self.disp[:, :, :, 1:, :] - self.disp[:, :, :, :-1, :])
        dz = torch.abs(self.disp[:, :, :, :, 1:] - self.disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class DiffeomorphicSTN3Dv2(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(DiffeomorphicSTN3Dv2, self).__init__()

        self.device = device
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        # Encoder-decoder
        dim = 3
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down4 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv23 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv3d(8, dim, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x5 = self.down4(x4)
        x5 = F.relu(self.conv51(x5))
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = F.relu(self.conv42(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='trilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='trilinear', align_corners=False)

        self.T = self.move_grid_dims(self.grid + self.disp)
        self.T_inv = self.move_grid_dims(self.grid + self.disp_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :, :] - self.disp[:, :, :-1, :, :])
        dx = torch.abs(self.disp[:, :, :, 1:, :] - self.disp[:, :, :, :-1, :])
        dz = torch.abs(self.disp[:, :, :, :, 1:] - self.disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class FullSTN3D(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(FullSTN3D, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 16))

        # Encoder-decoder
        dim = 3
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv23 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv3d(8, dim, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 3)
        self.rotation = nn.Linear(32, 3)
        self.scaling = nn.Linear(32, 3)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # predict affine parameters
        xa = F.avg_pool3d(x4, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='trilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='trilinear', align_corners=False)

        # compose transformations
        b, c, d, h, w = x.size()
        warp_field = self.grid + self.disp
        warp_field_inv = self.grid + self.disp_inv

        ones = torch.ones([b, 1, warp_field.view(b, 3, -1).size(2)]).to(self.device)

        # compose forward transformation
        grid_affine = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        grid_affine = torch.bmm(self.theta[:, 0:3, :], torch.cat((grid_affine.view(b, 3, -1), ones), dim=1))
        self.T = self.transform(warp_field, self.move_grid_dims(grid_affine.view(b, 3, d, h, w)))
        self.T = self.move_grid_dims(self.T)

        # compose backward transformation
        warp_field_inv = warp_field_inv.view(b, 3, -1)
        self.T_inv = torch.bmm(self.theta_inv[:, 0:3, :], torch.cat((warp_field_inv, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 3, d, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :, :] - self.disp[:, :, :-1, :, :])
        dx = torch.abs(self.disp[:, :, :, 1:, :] - self.disp[:, :, :, :-1, :])
        dz = torch.abs(self.disp[:, :, :, :, 1:] - self.disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # Affine transform
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp


class FullSTN3Dv2(nn.Module):
    def __init__(self, input_size, input_channels, device):
        super(FullSTN3Dv2, self).__init__()

        self.device = device
        self.input_size = input_size
        self.register_buffer('grid', self.get_normalized_grid(input_size[::-1]))

        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
        num_features = torch.prod(torch.floor_divide(torch.tensor(input_size), 32))

        # Encoder-decoder
        dim = 3
        use_bias = True
        self.down1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv21 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv22 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down2 = nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv31 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.down3 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv41 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.down4 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv51 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv42 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv32 = nn.Conv3d(48, 16, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv23 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv24 = nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=use_bias)
        self.conv25 = nn.Conv3d(8, dim, kernel_size=3, padding=1, bias=use_bias)

        # Affine part
        self.fc = nn.Linear(32 * num_features, 32)
        self.translation = nn.Linear(32, 3)
        self.rotation = nn.Linear(32, 3)
        self.scaling = nn.Linear(32, 3)

        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))
        self.scaling.weight.data.zero_()
        self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=self.dtype))

    def forward(self, x):
        x2 = self.down1(x)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        x3 = self.down2(x2)
        x3 = F.relu(self.conv31(x3))
        x4 = self.down3(x3)
        x4 = F.relu(self.conv41(x4))
        x5 = self.down4(x4)
        x5 = F.relu(self.conv51(x5))
        x4 = torch.cat([self.up4(x5), x4], dim=1)
        x4 = F.relu(self.conv42(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = F.relu(self.conv32(x3))
        x2 = torch.cat([self.up2(x3), x2], dim=1)
        x2 = F.relu(self.conv23(x2))
        x2 = F.relu(self.conv24(x2))

        # predict flow field
        flow = self.conv25(x2)

        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)
        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # predict affine parameters
        xa = F.avg_pool3d(x5, 2)
        xa = xa.view(xa.size(0), -1)
        xa = F.relu(self.fc(xa))
        self.affine_matrix(xa)

        # integrate velocity field
        self.disp = self.integrate(flow, 6)
        self.disp_inv = self.integrate(flow * -1, 6)

        self.disp = F.interpolate(self.disp, scale_factor=2, mode='trilinear', align_corners=False)
        self.disp_inv = F.interpolate(self.disp_inv, scale_factor=2, mode='trilinear', align_corners=False)

        # compose transformations
        b, c, d, h, w = x.size()
        warp_field = self.grid + self.disp
        warp_field_inv = self.grid + self.disp_inv

        ones = torch.ones([b, 1, warp_field.view(b, 3, -1).size(2)]).to(self.device)

        # compose forward transformation
        grid_affine = self.grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)
        grid_affine = torch.bmm(self.theta[:, 0:3, :], torch.cat((grid_affine.view(b, 3, -1), ones), dim=1))
        self.T = self.transform(warp_field, self.move_grid_dims(grid_affine.view(b, 3, d, h, w)))
        self.T = self.move_grid_dims(self.T)

        # compose backward transformation
        warp_field_inv = warp_field_inv.view(b, 3, -1)
        self.T_inv = torch.bmm(self.theta_inv[:, 0:3, :], torch.cat((warp_field_inv, ones), dim=1))
        self.T_inv = self.T_inv.view(b, 3, d, h, w)
        self.T_inv = self.move_grid_dims(self.T_inv)

        # extract first channel for warping
        img = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(img).to(self.device)

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()
        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid

    def integrate(self, vel, nb_steps):
        grid = self.get_normalized_grid(vel.shape[2:]).to(self.device)
        disp = vel / (2 ** nb_steps)
        for _ in range(nb_steps):
            warped_disp = self.transform(disp, self.move_grid_dims(grid + disp), padding='border')
            disp = disp + warped_disp
        return disp

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :, :] - self.disp[:, :, :-1, :, :])
        dx = torch.abs(self.disp[:, :, :, 1:, :] - self.disp[:, :, :, :-1, :])
        dz = torch.abs(self.disp[:, :, :, :, 1:] - self.disp[:, :, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

    def get_disp(self):
        return self.disp

    def get_disp_inv(self):
        return self.disp_inv

    def get_T(self):
        return self.T

    def get_T_inv(self):
        return self.T_inv

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)
        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)

    def affine_matrix(self, x):
        b = x.size(0)

        trans = torch.tanh(self.translation(x)) * 1.0
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
        angle_1 = rot[:, 0].view(-1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Z
        angle_3 = rot[:, 2].view(-1)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 2] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        scale = torch.tanh(self.scaling(x)) * 1.0
        scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float).to(self.device)
        scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        scaling_matrix[:, 3, 3] = 1.0

        # Affine transform
        matrix = torch.bmm(scaling_matrix, rotation_matrix)
        matrix = torch.bmm(matrix, translation_matrix)

        self.theta = matrix
        self.theta_inv = torch.inverse(matrix)

    def warp_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T, interpolation=interpolation, padding=padding)
        return wrp

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = self.transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp
