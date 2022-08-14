import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, criterion=nn.BCELoss(), is_logit=True, z_value=3):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.z_value = z_value
        self.is_logit = is_logit
        self.base_loss = criterion

    def clip_tensor(self, pred, is_min_clip, apply_clip=True):
        z_value = self.z_value
        if z_value > 0 and apply_clip:
            with torch.no_grad():
                if self.is_logit:
                    tensor_sigmoid = torch.sigmoid(pred)
                else:
                    tensor_sigmoid = pred
                std, mean = torch.std_mean(tensor_sigmoid, unbiased=False)
                min = mean - std * z_value
                max = mean + std * z_value
                if self.is_logit:
                    min = torch.logit(min)
                    max = torch.logit(max)

            if is_min_clip:
                pred[pred < min] += std * z_value
            else:
                pred[pred > max] -= std * z_value
        return pred

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def forward(self, pred, target_is_real, no_clip=True):
        return self.base_loss(self.clip_tensor(pred, target_is_real, not no_clip), self.get_target_tensor(pred, target_is_real))


