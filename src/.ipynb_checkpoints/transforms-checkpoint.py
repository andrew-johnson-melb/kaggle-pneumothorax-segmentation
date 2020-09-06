import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(norm_mean=[0.485, 0.456, 0.406] , norm_std=[0.229, 0.224, 0.225]):
    """Construct a list of transforms for the training and validation sets. 
    The transforms and their parameters used below are inspired (taken) from 
    kagglers.
    
    returns:
        training_transforms, validation_transforms
    
    """
    aug_training = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomContrast(),
            A.RandomGamma(),
            A.RandomBrightness()]
            , p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        A.ShiftScaleRotate(p=0.5, shift_limit=[-0.0625, 0.0625]),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    aug_validation = A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    return aug_training, aug_validation