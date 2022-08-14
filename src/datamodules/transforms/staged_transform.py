import torchvision.transforms as transforms
from torchvision.transforms import Lambda
import albumentations as A

class StagedTransform(transforms.Compose):


    def get_alb_transform(self, alb_transforms):
        composed = A.Compose(alb_transforms, p=1)
        alb_transform = [Lambda(lambda x: composed(image=x)['image'])]

        return transforms.Compose(alb_transform)


    def __init__(self, pre_transform, strong_transform=[], post_transform=[]):
        self.pre_transform = self.get_alb_transform(pre_transform)
        self.strong_transform = self.get_alb_transform(strong_transform)
        self.post_transform = self.get_alb_transform(post_transform)

        super().__init__([self.get_alb_transform(pre_transform + strong_transform + post_transform)])

