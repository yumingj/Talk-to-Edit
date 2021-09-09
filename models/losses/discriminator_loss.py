import torch
import torch.nn as nn
from models.archs.stylegan2.model import Discriminator
from torch.nn import functional as F


class DiscriminatorLoss(nn.Module):

    def __init__(self, pretrained_model, img_res):
        super(DiscriminatorLoss, self).__init__()
        if img_res == 128:
            self.discriminator = Discriminator(
                size=img_res, channel_multiplier=1)
            self.discriminator.load_state_dict(
                torch.load(pretrained_model)['d'], strict=True)
        elif img_res == 1024:
            self.discriminator = Discriminator(
                size=img_res, channel_multiplier=2)
            self.discriminator.load_state_dict(
                torch.load(pretrained_model), strict=True)
        self.discriminator.to(torch.device('cuda'))
        self.discriminator.eval()

    def forward(self, generated_images):
        generated_pred = self.discriminator(generated_images)
        loss = F.softplus(-generated_pred).mean()

        return loss
