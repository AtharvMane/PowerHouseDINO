import zarr
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import config as cfg


def get_image_and_segmentation_maps(group: da.array)-> tuple[np.array, np.array]:
    """
    Extracts a 2D mid-section slice from both EM imagery and mitochondrial 
    segmentation labels within a Dask-backed group.

    The function accesses full-resolution (s0) data, performs a spatial slice 
    along the y-axis at the center of the volume, and triggers a computation 
    to return the results as NumPy arrays.

    Parameters
    ----------
    group : da.array
        A Dask-compatible group or hierarchy containing the datasets 
        'labels/mito_seg/s0' and 'em/fibsem-uint16/s0'.

    Returns
    -------
    img_map : np.array
        A 2D NumPy array representing the central cross-section of the 
        electron microscopy (EM) volume.
    seg_map : np.array
        A 2D NumPy array representing the central cross-section of the 
        mitochondrial segmentation labels.

    Notes
    -----
    - This function assumes the input group structure follows a specific 
      hierarchy (e.g., an HDF5 or Zarr group).
    - The slicing is performed on the second dimension: `shape[1] // 2`.
    - Using `.compute()` will load the specific slice into memory; ensure 
      the slice dimensions are manageable for your available RAM.
    """
    seg_data = group['labels/mito_seg/s0']
    seg_ddata = da.from_array(seg_data, chunks = seg_data.chunks)
    img_data = group['em/fibsem-uint16/s0'] # s0 is the the full-resolution data for this particular volume
    img_ddata = da.from_array(img_data, chunks=img_data.chunks)
    img_map = img_ddata[:,img_ddata.shape[1]//2,:].compute()
    seg_map = seg_ddata[:,seg_ddata.shape[1]//2,:].compute()
    return img_map, seg_map


def get_control_regions(img: np.array, num_regions: int, size_:int):
    """
    Extracts a specified number of random square control regions (patches) from an image.

    This function generates random top-left coordinates and slices square patches 
    of a fixed size from the input image. It returns both the extracted pixel 
    data and the coordinates used for the extraction.

    Args:
        img (np.array): The input image array (usually 2D for grayscale or 3D for RGB).
        num_regions (int): The total number of random regions to extract.
        size_ (int): The side length (in pixels) of the square regions.

    Returns:
        tuple: A tuple containing:
            - control_slices (list of np.array): A list of the extracted image patches.
            - start_indices (np.array): An array of shape (num_regions, 2) containing 
              the [row, col] starting coordinates for each patch.

    Note:
        The function uses `np.random.random_integers`, which is technically 
        deprecated in newer NumPy versions. For modern code, consider 
        switching to `np.random.randint`.
    """
    start_indices = np.random.random_integers(low = 0, high = min(img.shape[0], img.shape[1])-size_-1, size=(num_regions, 2))
    control_slices = []
    for start_index in start_indices:
        control_slice = img[start_index[0]:start_index[0]+size_, start_index[1]:start_index[1]+size_]
        control_slice = control_slice[:,:,None]
        control_slice = np.concatenate([control_slice, control_slice, control_slice], axis = 2)
        control_slices.append(control_slice)
    return control_slices



def calc_bounds(min_, max_, size_, bound_max):
    """
    Calculates a centered window of a fixed size around a given range, 
    clamped within a maximum boundary.

    The function finds the midpoint between `min_` and `max_`, attempts to 
    center a window of length `size_` around that midpoint, and then shifts 
    the window if it overflows the boundaries (0 or `bound_max`).

    Args:
        min_ (int): The lower bound of the target range to center around.
        max_ (int): The upper bound of the target range to center around.
        size_ (int): The desired total length of the resulting window.
        bound_max (int): The absolute maximum limit for the upper boundary (e.g., image width).

    Returns:
        tuple: A tuple (min_new, max_new) representing the inclusive lower 
               and exclusive upper bounds of the clamped window.

    Raises:
        RuntimeError: If the requested `size_` is larger than the `bound_max`.
    """
    r_1 = (max_+min_-size_)//2
    min_new = max(0, r_1)
    max_new = min(bound_max, r_1+size_)

    if size_>bound_max:
        raise RuntimeError("Impossible Scenario, min_, max_ out of bounds")
    
    if min_new==0:
        return 0, size_
    elif max_new==bound_max:
        return bound_max-size_, bound_max
    else:
        return min_new, max_new


def get_final_data_list(img_map: np.array, seg_map: np.array, max_size : int = 256)->tuple[list[np.array], list[np.array]]:
    """
    Extracts and crops individual mitochondrial instances from a source image based on a segmentation map.

    This function iterates through every unique object ID in the segmentation map, calculates 
    its spatial bounds, and extracts a cropped slice from both the image and the mask. 
    The crops are adjusted via a helper function (`calc_bounds`) to ensure they meet 
    specific size constraints.

    Args:
        img_map (np.array): The source grayscale or multi-channel image of the biological sample.
        seg_map (np.array): An integer-labeled segmentation map where each unique 
            integer represents a different mitochondrial instance.
        max_size (int, optional): The target bounding box size for the crops. 
            Defaults to 256.

    Returns:
        tuple[list[np.array], list[np.array], list[int], list[int]]: 
            - mitochondria_image_slices: List of cropped image patches. Each patch is 
              converted to 3-channel (RGB) by concatenating the grayscale slice.
            - mitochondria_seg_slices: List of corresponding cropped segmentation masks.
            - heights: List of heights for each extracted crop.
            - widths: List of widths for each extracted crop.
    """
    mitochondria_image_slices = []
    mitochondria_seg_slices = []
    heights = []
    widths = []

    for i in tqdm(range(1,np.max(seg_map))):
        if np.all(seg_map!=i):
            continue
        poses = np.where(seg_map==i)
        min_x = np.min(poses[0])
        max_x = np.max(poses[0])
        min_y = np.min(poses[1])
        max_y = np.max(poses[1])
        min_x, max_x = calc_bounds(min_x, max_x, max_size, img_map.shape[0])
        min_y, max_y = calc_bounds(min_y, max_y, max_size, img_map.shape[1])

        heights.append(max_x-min_x)
        widths.append(max_y-min_y)
        img_slice = img_map[min_x:max_x, min_y:max_y, None]
        seg_slice = seg_map[min_x:max_x,min_y:max_y]

        img_slice = np.concatenate([img_slice, img_slice, img_slice], axis = 2)
        mitochondria_image_slices.append(img_slice)
        mitochondria_seg_slices.append(seg_slice)

    return mitochondria_image_slices, mitochondria_seg_slices, heights, widths


def plot_img_size_histograms(widths: list[int], heights: list[int]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(heights, bins=83, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Image Heights')
    axes[0].set_xlabel('Height (pixels)')
    axes[0].set_ylabel('Frequency')

    # Width Histogram
    axes[1].hist(widths, bins=83, color='salmon', edgecolor='black')
    axes[1].set_title('Distribution of Image Widths')
    axes[1].set_xlabel('Width (pixels)')
    axes[1].set_ylabel('Frequency') 
    plt.show()


if __name__=='__main__':

    
    for name, path in cfg.DATASET_LINKS.items():
        group = zarr.open(zarr.N5FSStore(path, anon=True))
        img_map, seg_map = get_image_and_segmentation_maps(group)

        mitochondria_image_slices, mitochondria_seg_slices, heights, widths = get_final_data_list(
            img_map, seg_map, max_size = cfg.DATASETS_IMAGE_SIZE
        )
        control_image_slices = get_control_regions(img_map, 80, size_=cfg.DATASETS_IMAGE_SIZE)
        np.save(f"./datasets/{name}_image_slices", mitochondria_image_slices)
        np.save(f"./datasets/{name}_control_image_slices", control_image_slices)
        np.save(f"./datasets/{name}_mitochondria_seg_slices", mitochondria_seg_slices)
