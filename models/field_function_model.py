import logging

import torch

from models.base_model import BaseModel

logger = logging.getLogger('base')


class FieldFunctionModel(BaseModel):

    def __init__(self, opt):
        super(FieldFunctionModel, self).__init__(opt)
        self.replaced_layers = opt['replaced_layers']
        self.fix_layers = True

    def modify_latent_code(self, latent_code_w, latent_code_w_plus=None):
        assert self.input_is_latent

        return_dict = {}
        # field function mapping
        field = self.field_function(latent_code_w)
        with torch.no_grad():
            offset_w = self.stylegan_gen.style_forward(
                torch.zeros_like(field), skip_norm=True)
        delta_w = self.stylegan_gen.style_forward(
            field, skip_norm=True) - offset_w

        if latent_code_w_plus is None:
            edited_latent_code = latent_code_w.unsqueeze(1).repeat(
                1, self.w_space_channel_num, 1)
        else:
            edited_latent_code = latent_code_w_plus.clone()
            return_dict['field'] = delta_w

        for layer_idx in range(self.replaced_layers):
            edited_latent_code[:, layer_idx, :] += delta_w

        return_dict['edited_latent_code'] = edited_latent_code
        return return_dict

    def modify_latent_code_bidirection(self,
                                       latent_code_w,
                                       latent_code_w_plus=None,
                                       alpha=1):
        assert self.input_is_latent

        return_dict = {}
        # field function mapping
        field = self.field_function(latent_code_w)
        with torch.no_grad():
            offset_w = self.stylegan_gen.style_forward(
                torch.zeros_like(field), skip_norm=True)
        delta_w = self.stylegan_gen.style_forward(
            field, skip_norm=True) - offset_w

        if latent_code_w_plus is None:
            edited_latent_code = latent_code_w.unsqueeze(1).repeat(
                1, self.w_space_channel_num, 1)
        else:
            edited_latent_code = latent_code_w_plus.clone()
            return_dict['field'] = delta_w

        for layer_idx in range(self.replaced_layers):
            edited_latent_code[:, layer_idx, :] += alpha * delta_w

        return_dict['edited_latent_code'] = edited_latent_code
        return return_dict
