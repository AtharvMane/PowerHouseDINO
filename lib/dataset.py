
import numpy as np
import torch

from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from typing import Any


def resize_image_for_patch_size(
        img: torch.tensor,
        upscaler: int | tuple[int, int]
)->torch.tensor:
    """
    Rescales an image to simulate a smaller transformer patch size.

    Transformers use fixed patch sizes that cannot be changed without altering 
    weight matrices. This function bypasses that limitation by upsampling the 
    input image by a factor of `upscaler`. By increasing the image resolution 
    while keeping the model's patch size constant, you effectively decrease 
    the "effective" patch size relative to the original image content.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        upscaler (int | tuple[int, int]): The magnification factor. If an int, 
            the same scale is applied to height and width. If a tuple, 
            specifies (height_scale, width_scale). 
            
            Note Upscaler = `original_transformer_patch_size//effective_patch_size`

    Returns:
        torch.Tensor: The upsampled image tensor with nearest-neighbor 
            interpolation to preserve pixel-level alignment.
            Shape: (C, H * upscaler[0], W * upscaler[1]).
    """
    c, h, w  = img.shape
    if isinstance(upscaler, int):
        img_new = TF.resize(img, (h*upscaler,w*upscaler),InterpolationMode.NEAREST)
    elif isinstance(upscaler, tuple):
        img_new = TF.resize(img, (h*upscaler[0],w*upscaler[1]),InterpolationMode.NEAREST)
    return img_new


class MitochondrialDataset(Dataset):
    """
    A PyTorch Dataset that loads NumPy arrays and rescales images to adjust
    the effective patch size for Transformer models.

    This dataset assumes the underlying Transformer has a fixed patch size 
    (defaulting to 16 based on the internal logic) and uses nearest-neighbor 
    upscaling to allow the model to process the image at a finer granularity.

    Attributes:
        data (torch.Tensor): The loaded dataset converted to a tensor.
        transforms (Any): Torchvision-style transformations to apply.
        upscaler (tuple[int, int]): The calculated height and width scaling factors.
    """
    def __init__(
            self,
            npy_dataset: np.array,
            transforms: Any,
            patch_size: int | tuple[int, int]
        )->None:
        super().__init__()
        if isinstance(patch_size, int):
            assert patch_size > 0, f"patch_size can't be <= 0, got patch_size = {patch_size}"
            self.upscaler = 16//patch_size, 16//patch_size
        elif isinstance(patch_size, tuple):
            assert patch_size[0] > 0 and patch_size>0, f"patch_size can't be <= 0, got patch_size = {patch_size}"
            self.upscaler = 16//patch_size[0], 16//patch_size[1]
        self.data = torch.tensor(npy_dataset)
        self.transforms = transforms
        
    
    def __len__(self)->None:
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[index].float()

        # Center the image pixel values
        img = (img-torch.min(img))/(torch.max(img)-torch.min(img))
        img = img.permute(2,0,1).contiguous()
        img = resize_image_for_patch_size(img, self.upscaler)
        if self.transforms is not None:
            img = self.transforms(img)
        return img