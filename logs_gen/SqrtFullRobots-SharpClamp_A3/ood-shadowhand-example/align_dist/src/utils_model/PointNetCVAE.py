import copy
import torch
import torch.nn as nn
import torch.utils.data
import math
import numpy as np


class PointNetEncoder(nn.Module):
    def __init__(self,
                 layers_size=[4, 64, 128, 512]):
        super(PointNetEncoder, self).__init__()
        self.layers_size = layers_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()

        for i in range(len(layers_size) - 1):
            self.conv_layers.append(nn.Conv1d(layers_size[i], layers_size[i + 1], 1))
            self.bn_layers.append(nn.BatchNorm1d(layers_size[i+1]))
            nn.init.xavier_normal_(self.conv_layers[-1].weight)

    def forward(self, x):
        # input: B * N * 4
        # output: B * latent_size
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers) - 1):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activate_func(x)
        x = self.bn_layers[-1](self.conv_layers[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.layers_size[-1])
        return x


class PointNetDecoder(nn.Module):
    def __init__(self,
                 global_feat_size=512,
                 latent_size=128,
                 pointwise_layers_size=[3, 64, 64],
                 global_layers_size=[64, 128, 512],
                 decoder_layers_size=[64+512+128, 512, 64, 64, 1]):
        super(PointNetDecoder, self).__init__()
        assert global_feat_size == global_layers_size[-1]
        assert decoder_layers_size[0] == latent_size + global_feat_size + pointwise_layers_size[-1]

        self.global_feat_size = global_feat_size
        self.latent_size = latent_size
        self.pointwise_layers_size = pointwise_layers_size
        self.global_layers_size = global_layers_size
        self.decoder_layers_size = decoder_layers_size

        self.pointwise_conv_layers = nn.ModuleList()
        self.pointwise_bn_layers = nn.ModuleList()
        self.global_conv_layers = nn.ModuleList()
        self.global_bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()

        for i in range(len(pointwise_layers_size) - 1):
            self.pointwise_conv_layers.append(nn.Conv1d(pointwise_layers_size[i], pointwise_layers_size[i + 1], 1))
            self.pointwise_bn_layers.append(nn.BatchNorm1d(pointwise_layers_size[i+1]))
            nn.init.xavier_normal_(self.pointwise_conv_layers[-1].weight)
        for i in range(len(global_layers_size) - 1):
            self.global_conv_layers.append(nn.Conv1d(global_layers_size[i], global_layers_size[i + 1], 1))
            self.global_bn_layers.append(nn.BatchNorm1d(global_layers_size[i+1]))
            nn.init.xavier_normal_(self.global_conv_layers[-1].weight)

        self.decoder_conv_layers = nn.ModuleList()
        self.decoder_bn_layers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        for i in range(len(decoder_layers_size) - 1):
            self.decoder_conv_layers.append(nn.Conv1d(decoder_layers_size[i], decoder_layers_size[i + 1], 1))
            self.decoder_bn_layers.append(nn.BatchNorm1d(decoder_layers_size[i + 1]))
            nn.init.xavier_normal_(self.decoder_conv_layers[-1].weight)

    def forward(self, x, z_latent_code):
        """
        :param x: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        bs = x.shape[0]
        npts = x.shape[1]

        pointwise_feature = x.transpose(1, 2)
        for i in range(len(self.pointwise_conv_layers) - 1):
            pointwise_feature = self.pointwise_conv_layers[i](pointwise_feature)
            pointwise_feature = self.pointwise_bn_layers[i](pointwise_feature)
            pointwise_feature = self.activate_func(pointwise_feature)
        pointwise_feature = self.pointwise_bn_layers[-1](self.pointwise_conv_layers[-1](pointwise_feature))

        global_feature = pointwise_feature.clone()
        for i in range(len(self.global_conv_layers) - 1):
            global_feature = self.global_conv_layers[i](global_feature)
            global_feature = self.global_bn_layers[i](global_feature)
            global_feature = self.activate_func(global_feature)
        global_feature = self.global_bn_layers[-1](self.global_conv_layers[-1](global_feature))
        global_feature = torch.max(global_feature, 2, keepdim=True)[0]
        global_feature = global_feature.view(bs, self.global_feat_size)

        global_feature = torch.cat([global_feature, z_latent_code], dim=1)
        global_feature = global_feature.view(bs, self.global_feat_size + self.latent_size, 1).repeat(1, 1, npts)
        pointwise_feature = torch.cat([pointwise_feature, global_feature], dim=1)
        for i in range(len(self.decoder_conv_layers) - 1):
            pointwise_feature = self.decoder_conv_layers[i](pointwise_feature)
            pointwise_feature = self.decoder_bn_layers[i](pointwise_feature)
            pointwise_feature = self.activate_func(pointwise_feature)
        pointwise_feature = self.decoder_bn_layers[-1](self.decoder_conv_layers[-1](pointwise_feature))

        # pointwise_feature: B x N x 1
        pointwise_feature = self.sigmoid(pointwise_feature).view(bs, npts)
        return pointwise_feature


class PointNetCVAE(nn.Module):
    def __init__(self,
                 latent_size=128,
                 encoder_layers_size=[4, 64, 128, 512],

                 decoder_global_feat_size=512,
                 decoder_pointwise_layers_size=[3, 64, 64],
                 decoder_global_layers_size=[64, 128, 512],
                 decoder_decoder_layers_size=[64+512+128, 512, 64, 64, 1]):
        super(PointNetCVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = PointNetEncoder(layers_size=encoder_layers_size)
        self.decoder = PointNetDecoder(latent_size=latent_size,
                                       global_feat_size=decoder_global_feat_size,
                                       pointwise_layers_size=decoder_pointwise_layers_size,
                                       global_layers_size=decoder_global_layers_size,
                                       decoder_layers_size=decoder_decoder_layers_size)

        self.encoder_z_means = nn.Linear(encoder_layers_size[-1], latent_size)
        self.encoder_z_logvars = nn.Linear(encoder_layers_size[-1], latent_size)

    def forward(self, object_cmap):
        """
        :param object_cmap: B x N x 4
        :return:
        """
        bs = object_cmap.shape[0]
        npts = object_cmap.shape[1]
        object_pts = object_cmap[:, :, :3].clone()
        means, logvars = self.forward_encoder(object_cmap=object_cmap)
        z_latent_code = self.reparameterize(means=means, logvars=logvars)
        cmap_values = self.forward_decoder(object_cmap[:, :, :3], z_latent_code).view(bs, npts)
        return cmap_values, means, logvars, z_latent_code

    def inference(self, object_pts, z_latent_code):
        """
        :param object_pts: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        cmap_values = self.forward_decoder(object_pts, z_latent_code)
        return cmap_values

    def reparameterize(self, means, logvars):
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        return means + eps * std

    def forward_encoder(self, object_cmap):
        cmap_feat = self.encoder(object_cmap)
        means = self.encoder_z_means(cmap_feat)
        logvars = self.encoder_z_logvars(cmap_feat)
        return means, logvars

    def forward_decoder(self, object_pts, z_latent_code):
        """
        :param object_pts: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        cmap_values = self.decoder(object_pts, z_latent_code)
        return cmap_values


if __name__ == '__main__':
    device = 'cuda'
    mode = 'test'
    model = PointNetCVAE()

    if mode == 'train':
        bs = 4
        model.train().cuda()
        dummy_object_camp = torch.randn(bs, 2048, 4, device=device).float()
        dummy_cmap_values, dummy_means, dummy_logvars, dummy_z = model(dummy_object_camp)

        loss_recons = torch.square(dummy_cmap_values - dummy_object_camp[:, :, 3]).mean()
        print(f'recons mean square error: {loss_recons}')
    elif mode == 'test':
        bs = 4
        model.eval().cuda()
        dummy_object_pts = torch.randn(bs, 2048, 3, device=device).float()
        z = torch.randn(bs, model.latent_size, device=device).float()
        dummy_cmap_values = model.inference(dummy_object_pts, z)
        print(f'dummy output size: {dummy_cmap_values.size()}')
        #
    else:
        raise NotImplementedError()


