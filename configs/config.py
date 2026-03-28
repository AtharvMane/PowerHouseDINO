import torch
from torchvision import transforms as T
from enum import Enum


class DistanceMetric(Enum):
    EUCLIDEAN_DISTANCE="euclidean_distance"
    COSINE_DISTANCE="cosine_distance"
    MANHATTEN_DISTANCE="manhatten_distance"


# Parameters for get_dataset.py
DATASET_LINKS = {
    "mito_jrc_jurkat": "s3://janelia-cosem-datasets/jrc_jurkat-1/jrc_jurkat-1.n5",
    "mito_jrc_hela-3": "s3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.n5",
    "mito_jrc_macrophage": "s3://janelia-cosem-datasets/jrc_macrophage-2/jrc_macrophage-2.n5"
}

DATASETS_IMAGE_SIZE = 256
NUM_CONTROL_IMAGES_PER_DATASET = 80


#Parameters for get_embeddings.py
DINOV3_PATH= "dinov3/"
DINOV3_WEIGHTS_PATH = "./weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
DINOV3_MODEL_TYPE = 'dinov3_vits16'

BATCH_SIZE = 1
EFFECTIVE_PATCH_SIZE = 4

TRANSFORMS =  T.Compose([
    T.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])


DISTANCE_METRIC = DistanceMetric.COSINE_DISTANCE

DEVICE = None
