import torch
from torch import nn
import torch.nn.functional as F

from src.models.components.losses.utils.box_utils import scale_tensor_box as scale_box


# scale for apply to scale face crop
# empty_scale to apply on image where crop is not presented, usually applied when gt image already scaled
class FaceCropper(nn.Module):
    def __init__(self, target_size, scale=1, empty_scale=1):
        super(FaceCropper, self).__init__()
        self.scale = scale
        self.target_size = target_size
        self.empty_scale = empty_scale

    def crop_face(self, x, crop=None):
        # TODO implement different sizes
        assert x.shape[2] == x.shape[3]

        DEFAULT_BBOX_SIZE = 256
        size = x.shape[2]


        if crop is None:
            # Resolution for 256
            y1, y2, x1, x2 = [coord * size / DEFAULT_BBOX_SIZE for coord in (35, 223, 32, 220)]
            x1, y1, x2, y2 = scale_box([x1, y1, x2, y2], self.scale / self.empty_scale)
            x = x[:, :, max(0, int(y1)):min(int(y2), size), max(0, int(x1)): min(int(x2), size)]  # Crop interesting region
            x = F.interpolate(x, (self.target_size, self.target_size), mode='bilinear')
        else:
            if isinstance(crop, list) or isinstance(crop, torch.Tensor) and len(crop.shape) == 1:
                x1, y1, x2, y2 = scale_box(crop, self.scale)
                x = x[:, :, max(0, int(y1)):min(int(y2), size), max(0, int(x1)): min(int(x2), size)]
                x = F.interpolate(x, (self.target_size, self.target_size), mode='bilinear', align_corners=False)
            else:
                results = []
                for i, crop_i in enumerate(crop):
                    x_i = x[i:i + 1]
                    if crop_i[0] >= crop_i[2] - 1 or crop_i[1] >= crop_i[3] - 1:
                        # no face found
                        # print(f'No face found for {i} crop')
                        y1, y2, x1, x2 = [coord * size / DEFAULT_BBOX_SIZE for coord in (35, 223, 32, 220)]
                        x1, y1, x2, y2 = scale_box([x1, y1, x2, y2], self.scale / self.empty_scale)
                        x_i = x_i[:, :, max(0, int(y1)):min(int(y2), size), max(0, int(x1)): min(int(x2), size)]  # Crop interesting region
                        x_i = F.interpolate(x_i, (self.target_size, self.target_size), mode='bilinear', align_corners=False)
                        results.append(x_i)
                    else:

                        x1, y1, x2, y2 = scale_box(crop_i, self.scale)
                        x_i = x_i[:, :, max(0, int(y1)):min(int(y2), size), max(0, int(x1)): min(int(x2), size)]
                        # print(x1, y1, x2, y2)
                        x_i = F.interpolate(x_i, (self.target_size, self.target_size), mode='bilinear', align_corners=False)
                        results.append(x_i)
                x = torch.cat(results, dim=0)
        return x

    def forward(self, x, crop=None):
        y_feats = self.crop_face(x, crop)  # Otherwise use the feature from there
        return y_feats
