import torch
from torch import nn
import torch.nn.functional as F


# From https://github.com/orpatashnik/StyleCLIP
from src.models.components.facial_recognition.model_irse import Backbone
from src.models.components.losses.face_cropper import FaceCropper

# From https://github.com/orpatashnik/StyleCLIP/blob/05f53df9514dd18a9c86195b68c7ceaf2b86d4f6/criteria/id_loss.py#L7
class IDLoss(nn.Module):
    def __init__(self, ir_se50_weights, empty_scale=1):
        super(IDLoss, self).__init__()
        self.scale = empty_scale
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se', apply_norm=False)
        self.facenet.load_state_dict(torch.load(ir_se50_weights))

        self.face_crop = FaceCropper(112, empty_scale=empty_scale)
        self.facenet.eval()


    def extract_feats(self, x, crop=None):
        x = self.face_crop(x, crop)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, crop=None):
        self.facenet.eval()
        n_samples = y.shape[0]
        with torch.no_grad():
            y_feats = self.extract_feats(y, crop)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat, crop)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count


class IDRMSELoss(IDLoss):
    def forward(self, y_hat, y, crop=None):
        self.facenet.eval()
        with torch.no_grad():
            y_feats = self.extract_feats(y, crop)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat, crop)
        y_feats = y_feats.detach()

        rmse_loss = torch.sqrt(F.mse_loss(y_hat_feats, y_feats, reduction='none').sum(1))
        return rmse_loss.mean()

class IDMSELoss(IDLoss):
    def forward(self, y_hat, y, crop=None):
        self.facenet.eval()
        with torch.no_grad():
            y_feats = self.extract_feats(y, crop)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat, crop)
        y_feats = y_feats.detach()

        mse_loss = F.mse_loss(y_hat_feats, y_feats)
        return mse_loss.mean()