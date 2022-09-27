import torch
import torch.nn as nn


class VAECriterion(nn.Module):
    def __init__(self, lw_init_recon, lw_init_kld, ann_temp, batchsize,
                       ann_per_epochs=1):
        super(VAECriterion, self).__init__()
        self.lw_recon = lw_init_recon
        self.lw_kld = lw_init_kld
        self.ann_temp = ann_temp
        self.batchsize = batchsize

        self.iter_counter = 0
        self.ann_per_epochs = ann_per_epochs

    def forward(self, means, logvars, cmap_values_gt, cmap_values_hat):
        """
        :param means:
        :param logvars:
        :param cmap_values_gt: B x N
        :param cmap_values_hat: B x N
        :return:
        """
        bs = cmap_values_gt.shape[0]
        npts = cmap_values_gt.shape[1]
        cmap_values_gt = cmap_values_gt.view(bs, npts)
        cmap_values_hat = cmap_values_hat.view(bs, npts)
        loss_kld = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp(), dim=-1).mean()
        loss_recon = torch.sqrt(torch.square(cmap_values_gt - cmap_values_hat).mean())

        loss = self.lw_kld * loss_kld + self.lw_recon * loss_recon

        return loss, loss_recon, loss_kld

    @staticmethod
    def metrics(self, cmap_values_gt, cmap_values_hat):
        return torch.sqrt(torch.square(cmap_values_gt - cmap_values_hat).mean())

    def apply_iter(self):
        if self.iter_counter % self.ann_per_epochs == self.ann_per_epochs - 1:
            self.lw_kld *= self.ann_temp
        self.iter_counter += 1


class VAEAttnCriterion(nn.Module):
    def __init__(self, lw_init_recon, lw_init_kld, ann_temp, batchsize,
                       ann_per_epochs=1, alpha=3.):
        # a_i = exp(cmap_value * alpha)
        super(VAEAttnCriterion, self).__init__()
        self.lw_recon = lw_init_recon
        self.lw_kld = lw_init_kld
        self.ann_temp = ann_temp
        self.alpha = alpha
        self.batchsize = batchsize

        self.iter_counter = 0
        self.ann_per_epochs = ann_per_epochs

    def forward(self, means, logvars, cmap_values_gt, cmap_values_hat):
        """
        :param means:
        :param logvars:
        :param cmap_values_gt: B x N
        :param cmap_values_hat: B x N
        :return:
        """
        bs = cmap_values_gt.shape[0]
        npts = cmap_values_gt.shape[1]
        cmap_values_gt = cmap_values_gt.view(bs, npts)
        cmap_values_hat = cmap_values_hat.view(bs, npts)
        loss_kld = -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp(), dim=-1).mean()
        square_error = torch.square(cmap_values_gt - cmap_values_hat)
        attention_weights = torch.exp(cmap_values_gt * self.alpha)
        square_error = square_error * attention_weights
        loss_recon = torch.sqrt(square_error.sum(dim=1) / attention_weights.sum(dim=1)).mean()

        loss = self.lw_kld * loss_kld + self.lw_recon * loss_recon

        return loss, loss_recon, loss_kld

    @staticmethod
    def metrics(self, cmap_values_gt, cmap_values_hat):
        square_error = torch.square(cmap_values_gt - cmap_values_hat)
        attention_weights = torch.exp(cmap_values_gt * self.alpha)
        square_error = square_error * attention_weights
        loss_recon = torch.sqrt(square_error.sum(dim=1) / attention_weights.sum(dim=1)).mean()
        return loss_recon

    def apply_iter(self):
        if self.iter_counter % self.ann_per_epochs == self.ann_per_epochs - 1:
            self.lw_kld *= self.ann_temp
        self.iter_counter += 1


if __name__ == '__main__':
    bs = 16
    npts = 2048
    criterion = VAEAttnCriterion(lw_init_recon=1., lw_init_kld=1., ann_temp=1., batchsize=bs,
                                 ann_per_epochs=1, alpha=3.)
    dummy_means = torch.randn(bs)
    dummy_logvars = torch.randn(bs)
    dummy_cmap_values_gt = torch.rand(bs, npts)
    dummy_cmap_values_hat = torch.rand(bs, npts)
    criterion(dummy_means, dummy_logvars, dummy_cmap_values_gt, dummy_cmap_values_hat)

