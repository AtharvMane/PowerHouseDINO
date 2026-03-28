import numpy as np
import torch

from lib.dataset import MitochondrialDataset

from torch.utils.data import DataLoader
from typing import Any
import configs.config as cfg
from tqdm import tqdm
from torch import nn
from matplotlib import pyplot as plt
import os


def get_model(
        model_type: str,
        model_repo_path: str,
        model_weights_path: str
    )->nn.Module:
    """
    Returns a version of DINOv3 Mode;
    """
    dinov3_vits16 = torch.hub.load(
        model_repo_path,
        model_type,
        source='local',
        weights=model_weights_path).to("mps")
    dinov3_vits16.eval()
    return dinov3_vits16


def get_data_loader(
        path: str,
        patch_size:int,
        transforms, 
        batch_size: int=4
    )-> torch.utils.data.DataLoader:
    """
    Return a simple torch dataloader after creating the datasets
    """
    dataset_ = np.load(path)
    mito_dataset_ = MitochondrialDataset(dataset_, transforms=transforms, patch_size=patch_size)
    data_loader_ = DataLoader(mito_dataset_, batch_size=batch_size, pin_memory=True, shuffle=False)
    return data_loader_


def get_dataset_embeddings(model, dataloader, device):
    """
    Computes and return the patch and cls embeddings.
    """
    dataset_patch_tokens = []
    dataset_cls_tokens = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, img in tqdm(enumerate(dataloader)):
            img = img.to(device)
            features = model.forward_features(img)
            patch_tokens = features['x_norm_patchtokens']
            cls_tokens = features['x_norm_clstoken']
            dataset_cls_tokens.append(cls_tokens)
            dataset_patch_tokens.append(patch_tokens)
    return {
        "cls_tokens": torch.cat(dataset_cls_tokens).cpu(),
        "patch_tokens": torch.cat(dataset_patch_tokens).cpu()
    }


def get_distances(
        query_embedding: torch.tensor,
        comparison_embeddings: torch.tensor,
        type:cfg.DistanceMetric = cfg.DistanceMetric.COSINE_DISTANCE
    )->torch.tensor:
    """
    Calculates the distance between a single query embedding and a set of comparison embeddings.

    Supported metrics include Cosine Distance, Euclidean Distance (L2), and 
    Manhattan Distance (L1).

    Args:
        query_embedding (torch.Tensor): A 1D tensor representing the single vector 
            to compare. Shape: (D,) or (1, D).
        comparison_embeddings (torch.Tensor): A 2D tensor representing the collection 
            of vectors to measure against. Shape: (N, D).
        type (cfg.DistanceMetric, optional): The distance metric to apply. 
            Defaults to cfg.DistanceMetric.COSINE_DISTANCE.

    Returns:
        torch.tensor: A 1D tensor containing the calculated distances between the 
            query and each comparison embedding. Shape: (N,).

    Raises:
        AttributeError: If an unsupported distance metric is provided.
    """

    if type == cfg.DistanceMetric.COSINE_DISTANCE:
        distances = 1-torch.matmul(
            query_embedding/query_embedding.norm(),
            (comparison_embeddings/comparison_embeddings.norm(dim=1)[:,None]).t()
        )[0]
    elif type == cfg.DistanceMetric.EUCLIDEAN_DISTANCE:
        distances = (comparison_embeddings-query_embedding).norm(dim=1)
    elif type == cfg.DistanceMetric.MANHATTEN_DISTANCE:
        distances = (comparison_embeddings-query_embedding).abs().sum(dim=1)

    return distances


def show_distance_graphs(distances, distance_metric):
    """Given a dictionary of datasets: distnaces, creates a single plot for all datastes"""
    plt.figure(figsize=(8, 5))
    for name, distance in distances.items():
        x = torch.arange(len(distance))
        plt.plot(x, distance.sort()[0], label=f"{name}", linestyle='-')
    plt.title("Comparison of Three NumPy Arrays")
    plt.xlabel("Embedding Index (Sorted w.r.t distance)")
    plt.ylabel(f"{distance_metric}")
    plt.legend() # This displays the labels defined in plt.plot()
    plt.grid(True)
    plt.savefig(f"results/{distance_metric}.png", bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    if cfg.DEVICE is not None:
        device = cfg.DEVICE
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = get_model(
        model_repo_path = cfg.DINOV3_PATH,
        model_type= cfg.DINOV3_MODEL_TYPE,
        model_weights_path = cfg.DINOV3_WEIGHTS_PATH
    )

    for name, path in cfg.DATASET_LINKS.items():
        dataset_path = f"./datasets/{name}_image_slices.npy"
        dataset_control_path = f"./datasets/{name}_control_image_slices.npy"

        if os.path.isfile(f"./embeddings/{name}.pth") and os.path.isfile(f"./embeddings/{name}_control.pth"):
            continue


        data_loader = get_data_loader(
            path = dataset_path,
            patch_size = cfg.EFFECTIVE_PATCH_SIZE,
            transforms = cfg.TRANSFORMS,
            batch_size = cfg.BATCH_SIZE
        )

        data_loader_control = get_data_loader(
            path = dataset_control_path,
            patch_size = cfg.EFFECTIVE_PATCH_SIZE,
            transforms = cfg.TRANSFORMS,
            batch_size = cfg.BATCH_SIZE
        )

        embeddings = get_dataset_embeddings(
            model = model, 
            dataloader = data_loader,
            device=device
        )

        embeddings_control = get_dataset_embeddings(
            model = model, 
            dataloader = data_loader_control,
            device=device
        )

        torch.save(embeddings,f"./embeddings/{name}.pth")
        torch.save(embeddings_control,f"./embeddings/{name}_control.pth")


    embeddings = {}
    for name, _ in cfg.DATASET_LINKS.items():
        embeddings[name] = torch.load(f"embeddings/{name}.pth")
        embeddings[f"{name}_control"] = torch.load(f"embeddings/{name}_control.pth")

    query_vector = embeddings[list(embeddings.keys())[0]]['cls_tokens'][0]

    distances = {}
    for dataset_name_i, dataset_embeddings_i in embeddings.items():
        distances[dataset_name_i] = get_distances(
            query_vector[None],
            dataset_embeddings_i['cls_tokens'],
            type = cfg.DISTANCE_METRIC
        )
    

    show_distance_graphs(distances, distance_metric=cfg.DISTANCE_METRIC)