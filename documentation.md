# `PowerHouseDINO`

## Table of Contents

- 🅼 [PowerHouseDINO](#PowerHouseDINO)
- 🅼 [PowerHouseDINO\.config](#PowerHouseDINO-config)
- 🅼 [PowerHouseDINO\.dataset](#PowerHouseDINO-dataset)
- 🅼 [PowerHouseDINO\.get\_dataset](#PowerHouseDINO-get_dataset)
- 🅼 [PowerHouseDINO\.get\_embeddings](#PowerHouseDINO-get_embeddings)

<a name="PowerHouseDINO"></a>
## 🅼 PowerHouseDINO
<a name="PowerHouseDINO-config"></a>
## 🅼 PowerHouseDINO\.config

- **Constants:**
  - 🆅 [DATASET\_LINKS](#PowerHouseDINO-config-DATASET_LINKS)
  - 🆅 [DATASETS\_IMAGE\_SIZE](#PowerHouseDINO-config-DATASETS_IMAGE_SIZE)
  - 🆅 [NUM\_CONTROL\_IMAGES\_PER\_DATASET](#PowerHouseDINO-config-NUM_CONTROL_IMAGES_PER_DATASET)
  - 🆅 [DINOV3\_PATH](#PowerHouseDINO-config-DINOV3_PATH)
  - 🆅 [DINOV3\_WEIGHTS\_PATH](#PowerHouseDINO-config-DINOV3_WEIGHTS_PATH)
  - 🆅 [DINOV\_MODEL\_TYPE](#PowerHouseDINO-config-DINOV_MODEL_TYPE)
  - 🆅 [BATCH\_SIZE](#PowerHouseDINO-config-BATCH_SIZE)
  - 🆅 [EFFECTIVE\_PATCH\_SIZE](#PowerHouseDINO-config-EFFECTIVE_PATCH_SIZE)
  - 🆅 [TRANSFORMS](#PowerHouseDINO-config-TRANSFORMS)
  - 🆅 [DISTANCE\_METRIC](#PowerHouseDINO-config-DISTANCE_METRIC)
  - 🆅 [DEVICE](#PowerHouseDINO-config-DEVICE)
- **Classes:**
  - 🅲 [DistanceMetric](#PowerHouseDINO-config-DistanceMetric)

### Constants

<a name="PowerHouseDINO-config-DATASET_LINKS"></a>
### 🆅 PowerHouseDINO\.config\.DATASET\_LINKS

```python
DATASET_LINKS = {'mito_jrc_jurkat': 's3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.n5', 'mito_jrc_hela-3': 's3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.n5', 'mito_jrc_macrophage': 's3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5'}
```
<a name="PowerHouseDINO-config-DATASETS_IMAGE_SIZE"></a>
### 🆅 PowerHouseDINO\.config\.DATASETS\_IMAGE\_SIZE

```python
DATASETS_IMAGE_SIZE = 256
```
<a name="PowerHouseDINO-config-NUM_CONTROL_IMAGES_PER_DATASET"></a>
### 🆅 PowerHouseDINO\.config\.NUM\_CONTROL\_IMAGES\_PER\_DATASET

```python
NUM_CONTROL_IMAGES_PER_DATASET = 80
```
<a name="PowerHouseDINO-config-DINOV3_PATH"></a>
### 🆅 PowerHouseDINO\.config\.DINOV3\_PATH

```python
DINOV3_PATH = 'dinov3/'
```
<a name="PowerHouseDINO-config-DINOV3_WEIGHTS_PATH"></a>
### 🆅 PowerHouseDINO\.config\.DINOV3\_WEIGHTS\_PATH

```python
DINOV3_WEIGHTS_PATH = './dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
```
<a name="PowerHouseDINO-config-DINOV_MODEL_TYPE"></a>
### 🆅 PowerHouseDINO\.config\.DINOV\_MODEL\_TYPE

```python
DINOV_MODEL_TYPE = 'dinov3_vits16'
```
<a name="PowerHouseDINO-config-BATCH_SIZE"></a>
### 🆅 PowerHouseDINO\.config\.BATCH\_SIZE

```python
BATCH_SIZE = 1
```
<a name="PowerHouseDINO-config-EFFECTIVE_PATCH_SIZE"></a>
### 🆅 PowerHouseDINO\.config\.EFFECTIVE\_PATCH\_SIZE

```python
EFFECTIVE_PATCH_SIZE = 4
```
<a name="PowerHouseDINO-config-TRANSFORMS"></a>
### 🆅 PowerHouseDINO\.config\.TRANSFORMS

```python
TRANSFORMS = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```
<a name="PowerHouseDINO-config-DISTANCE_METRIC"></a>
### 🆅 PowerHouseDINO\.config\.DISTANCE\_METRIC

```python
DISTANCE_METRIC = DistanceMetric.MANHATTEN_DISTANCE
```
<a name="PowerHouseDINO-config-DEVICE"></a>
### 🆅 PowerHouseDINO\.config\.DEVICE

```python
DEVICE = 'mps'
```

### Classes

<a name="PowerHouseDINO-config-DistanceMetric"></a>
### 🅲 PowerHouseDINO\.config\.DistanceMetric

```python
class DistanceMetric(Enum):
```
<a name="PowerHouseDINO-dataset"></a>
## 🅼 PowerHouseDINO\.dataset

- **Functions:**
  - 🅵 [resize\_image\_for\_patch\_size](#PowerHouseDINO-dataset-resize_image_for_patch_size)
- **Classes:**
  - 🅲 [MitochondrialDataset](#PowerHouseDINO-dataset-MitochondrialDataset)

### Functions

<a name="PowerHouseDINO-dataset-resize_image_for_patch_size"></a>
### 🅵 PowerHouseDINO\.dataset\.resize\_image\_for\_patch\_size

```python
def resize_image_for_patch_size(img: torch.tensor, upscaler: int | tuple[int, int]) -> torch.tensor:
```

Rescales an image to simulate a smaller transformer patch size\.

Transformers use fixed patch sizes that cannot be changed without altering 
weight matrices\. This function bypasses that limitation by upsampling the 
input image by a factor of \`upscaler\`\. By increasing the image resolution 
while keeping the model's patch size constant, you effectively decrease 
the "effective" patch size relative to the original image content\.

**Parameters:**

- **img** (`torch.Tensor`): Input image tensor of shape \(C, H, W\)\.
- **upscaler** (`int | tuple[int, int]`): The magnification factor\. If an int, 
the same scale is applied to height and width\. If a tuple, 
specifies \(height\_scale, width\_scale\)\. 

Note Upscaler = \`original\_transformer\_patch\_size//effective\_patch\_size\`

**Returns:**

- `torch.Tensor`: The upsampled image tensor with nearest-neighbor 
interpolation to preserve pixel-level alignment\.
Shape: \(C, H \* upscaler\[0\], W \* upscaler\[1\]\)\.

### Classes

<a name="PowerHouseDINO-dataset-MitochondrialDataset"></a>
### 🅲 PowerHouseDINO\.dataset\.MitochondrialDataset

```python
class MitochondrialDataset(Dataset):
```

A PyTorch Dataset that loads NumPy arrays and rescales images to adjust

the effective patch size for Transformer models\.

This dataset assumes the underlying Transformer has a fixed patch size 
\(defaulting to 16 based on the internal logic\) and uses nearest-neighbor 
upscaling to allow the model to process the image at a finer granularity\.

**Attributes:**

- **data** (`torch.Tensor`): The loaded dataset converted to a tensor\.
- **transforms** (`Any`): Torchvision-style transformations to apply\.
- **upscaler** (`tuple[int, int]`): The calculated height and width scaling factors\.

**Functions:**

<a name="PowerHouseDINO-dataset-MitochondrialDataset-__init__"></a>
#### 🅵 PowerHouseDINO\.dataset\.MitochondrialDataset\.\_\_init\_\_

```python
def __init__(self, npy_dataset: np.array, transforms: Any, patch_size: int | tuple[int, int]) -> None:
```
<a name="PowerHouseDINO-dataset-MitochondrialDataset-__len__"></a>
#### 🅵 PowerHouseDINO\.dataset\.MitochondrialDataset\.\_\_len\_\_

```python
def __len__(self) -> None:
```
<a name="PowerHouseDINO-dataset-MitochondrialDataset-__getitem__"></a>
#### 🅵 PowerHouseDINO\.dataset\.MitochondrialDataset\.\_\_getitem\_\_

```python
def __getitem__(self, index):
```
<a name="PowerHouseDINO-get_dataset"></a>
## 🅼 PowerHouseDINO\.get\_dataset

- **Functions:**
  - 🅵 [get\_image\_and\_segmentation\_maps](#PowerHouseDINO-get_dataset-get_image_and_segmentation_maps)
  - 🅵 [get\_control\_regions](#PowerHouseDINO-get_dataset-get_control_regions)
  - 🅵 [calc\_bounds](#PowerHouseDINO-get_dataset-calc_bounds)
  - 🅵 [get\_final\_data\_list](#PowerHouseDINO-get_dataset-get_final_data_list)
  - 🅵 [plot\_img\_size\_histograms](#PowerHouseDINO-get_dataset-plot_img_size_histograms)

### Functions

<a name="PowerHouseDINO-get_dataset-get_image_and_segmentation_maps"></a>
### 🅵 PowerHouseDINO\.get\_dataset\.get\_image\_and\_segmentation\_maps

```python
def get_image_and_segmentation_maps(group: da.array) -> tuple[np.array, np.array]:
```

Extracts a 2D mid-section slice from both EM imagery and mitochondrial

segmentation labels within a Dask-backed group\.

The function accesses full-resolution \(s0\) data, performs a spatial slice 
along the y-axis at the center of the volume, and triggers a computation 
to return the results as NumPy arrays\.

**Parameters:**

- **group** (`da.array`): A Dask-compatible group or hierarchy containing the datasets 
'labels/mito\_seg/s0' and 'em/fibsem-uint16/s0'\.

**Returns:**

- `np.array`: A 2D NumPy array representing the central cross-section of the 
electron microscopy \(EM\) volume\.
<a name="PowerHouseDINO-get_dataset-get_control_regions"></a>
### 🅵 PowerHouseDINO\.get\_dataset\.get\_control\_regions

```python
def get_control_regions(img: np.array, num_regions: int, size_: int):
```

Extracts a specified number of random square control regions \(patches\) from an image\.

This function generates random top-left coordinates and slices square patches 
of a fixed size from the input image\. It returns both the extracted pixel 
data and the coordinates used for the extraction\.

**Parameters:**

- **img** (`np.array`): The input image array \(usually 2D for grayscale or 3D for RGB\)\.
- **num_regions** (`int`): The total number of random regions to extract\.
- **size_** (`int`): The side length \(in pixels\) of the square regions\.

**Returns:**

- `tuple`: A tuple containing:
- control\_slices \(list of np\.array\): A list of the extracted image patches\.
- start\_indices \(np\.array\): An array of shape \(num\_regions, 2\) containing 
  the \[row, col\] starting coordinates for each patch\.
<a name="PowerHouseDINO-get_dataset-calc_bounds"></a>
### 🅵 PowerHouseDINO\.get\_dataset\.calc\_bounds

```python
def calc_bounds(min_, max_, size_, bound_max):
```

Calculates a centered window of a fixed size around a given range,

clamped within a maximum boundary\.

The function finds the midpoint between \`min\_\` and \`max\_\`, attempts to 
center a window of length \`size\_\` around that midpoint, and then shifts 
the window if it overflows the boundaries \(0 or \`bound\_max\`\)\.

**Parameters:**

- **min_** (`int`): The lower bound of the target range to center around\.
- **max_** (`int`): The upper bound of the target range to center around\.
- **size_** (`int`): The desired total length of the resulting window\.
- **bound_max** (`int`): The absolute maximum limit for the upper boundary \(e\.g\., image width\)\.

**Returns:**

- `tuple`: A tuple \(min\_new, max\_new\) representing the inclusive lower 
and exclusive upper bounds of the clamped window\.

**Raises:**

- **RuntimeError**: If the requested \`size\_\` is larger than the \`bound\_max\`\.
<a name="PowerHouseDINO-get_dataset-get_final_data_list"></a>
### 🅵 PowerHouseDINO\.get\_dataset\.get\_final\_data\_list

```python
def get_final_data_list(img_map: np.array, seg_map: np.array, max_size: int = 256) -> tuple[list[np.array], list[np.array]]:
```

Extracts and crops individual mitochondrial instances from a source image based on a segmentation map\.

This function iterates through every unique object ID in the segmentation map, calculates 
its spatial bounds, and extracts a cropped slice from both the image and the mask\. 
The crops are adjusted via a helper function \(\`calc\_bounds\`\) to ensure they meet 
specific size constraints\.

**Parameters:**

- **img_map** (`np.array`): The source grayscale or multi-channel image of the biological sample\.
- **seg_map** (`np.array`): An integer-labeled segmentation map where each unique 
integer represents a different mitochondrial instance\.
- **max_size** (`int`): The target bounding box size for the crops\. 
Defaults to 256\.

**Returns:**

- `tuple[list[np.array], list[np.array], list[int], list[int]]`: - mitochondria\_image\_slices: List of cropped image patches\. Each patch is 
  converted to 3-channel \(RGB\) by concatenating the grayscale slice\.
- mitochondria\_seg\_slices: List of corresponding cropped segmentation masks\.
- heights: List of heights for each extracted crop\.
- widths: List of widths for each extracted crop\.
<a name="PowerHouseDINO-get_dataset-plot_img_size_histograms"></a>
### 🅵 PowerHouseDINO\.get\_dataset\.plot\_img\_size\_histograms

```python
def plot_img_size_histograms(widths: list[int], heights: list[int]):
```
<a name="PowerHouseDINO-get_embeddings"></a>
## 🅼 PowerHouseDINO\.get\_embeddings

- **Functions:**
  - 🅵 [get\_model](#PowerHouseDINO-get_embeddings-get_model)
  - 🅵 [get\_data\_loader](#PowerHouseDINO-get_embeddings-get_data_loader)
  - 🅵 [get\_dataset\_embeddings](#PowerHouseDINO-get_embeddings-get_dataset_embeddings)
  - 🅵 [get\_distances](#PowerHouseDINO-get_embeddings-get_distances)
  - 🅵 [show\_distance\_graphs](#PowerHouseDINO-get_embeddings-show_distance_graphs)

### Functions

<a name="PowerHouseDINO-get_embeddings-get_model"></a>
### 🅵 PowerHouseDINO\.get\_embeddings\.get\_model

```python
def get_model(model_type: str, model_repo_path: str, model_weights_path: str) -> nn.Module:
```

Returns a version of DINOv3 Mode;
<a name="PowerHouseDINO-get_embeddings-get_data_loader"></a>
### 🅵 PowerHouseDINO\.get\_embeddings\.get\_data\_loader

```python
def get_data_loader(path: str, patch_size: int, transforms, batch_size: int = 4) -> torch.utils.data.DataLoader:
```

Return a simple torch dataloader after creating the datasets
<a name="PowerHouseDINO-get_embeddings-get_dataset_embeddings"></a>
### 🅵 PowerHouseDINO\.get\_embeddings\.get\_dataset\_embeddings

```python
def get_dataset_embeddings(model, dataloader, device):
```

Computes and return the patch and cls embeddings\.
<a name="PowerHouseDINO-get_embeddings-get_distances"></a>
### 🅵 PowerHouseDINO\.get\_embeddings\.get\_distances

```python
def get_distances(query_embedding: torch.tensor, comparison_embeddings: torch.tensor, type: cfg.DistanceMetric = cfg.DistanceMetric.COSINE_DISTANCE) -> torch.tensor:
```

Calculates the distance between a single query embedding and a set of comparison embeddings\.

Supported metrics include Cosine Distance, Euclidean Distance \(L2\), and 
Manhattan Distance \(L1\)\.

**Parameters:**

- **query_embedding** (`torch.Tensor`): A 1D tensor representing the single vector 
to compare\. Shape: \(D,\) or \(1, D\)\.
- **comparison_embeddings** (`torch.Tensor`): A 2D tensor representing the collection 
of vectors to measure against\. Shape: \(N, D\)\.
- **type** (`cfg.DistanceMetric`): The distance metric to apply\. 
Defaults to cfg\.DistanceMetric\.COSINE\_DISTANCE\.

**Returns:**

- `torch.tensor`: A 1D tensor containing the calculated distances between the 
query and each comparison embedding\. Shape: \(N,\)\.

**Raises:**

- **AttributeError**: If an unsupported distance metric is provided\.
<a name="PowerHouseDINO-get_embeddings-show_distance_graphs"></a>
### 🅵 PowerHouseDINO\.get\_embeddings\.show\_distance\_graphs

```python
def show_distance_graphs(distances, distance_metric):
```

Given a dictionary of datasets: distnaces, creates a single plot for all datastes
