import segmentation_models_pytorch as smp
from torch import nn
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def fix_padding(model: nn.Module, padding='reflect'):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            print(module)
            module.padding_mode = padding

class SegmentationModelUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=3):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=out_channels,  # model output channels (number of classes in your dataset)
        )
        fix_padding(self.model)

    def forward(self, x):
        return self.model(x)

    @classmethod
    def get_from_params(cls, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=3):
        return SegmentationModelUNet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, out_channels=out_channels)
