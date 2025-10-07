import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import trimesh
import urllib.request
import zipfile
from .shapenet_data_pc import ShapeNet15kPointClouds

class ShapeNetV2Dataset(Dataset):
    def __init__(self, root_dir, categories=None, num_points=2048, split='train'):
        self.root_dir = root_dir
        self.num_points = num_points
        self.split = split

        # Download and extract the dataset if it doesn't exist
        if not os.path.exists(root_dir):
            self.download_shapenet(root_dir)

        # Load the taxonomy file
        with open(os.path.join(root_dir, 'taxonomy.json'), 'r') as f:
            self.taxonomy = json.load(f)

        # Filter categories if specified
        if categories is not None:
            self.taxonomy = [t for t in self.taxonomy if t['synsetId'] in categories]

        # Load the split file
        split_file = os.path.join(root_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            self.models = [line.strip() for line in f]

        self.category_to_synsetId = {t['name']: t['synsetId'] for t in self.taxonomy}

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model = self.models[idx]
        synsetId, model_id = model.split('-')

        # Load the point cloud
        mesh_path = os.path.join(self.root_dir, synsetId, model_id, 'models', 'model_normalized.obj')
        mesh = trimesh.load(mesh_path)
        points = mesh.sample(self.num_points)

        return torch.FloatTensor(points)

    @staticmethod
    def download_shapenet(root_dir):
        url = "https://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v2.zip"
        zip_path = os.path.join(root_dir, "ShapeNetCore.v2.zip")

        # Create the directory if it doesn't exist
        os.makedirs(root_dir, exist_ok=True)

        # Download the dataset
        print("Downloading ShapeNetV2 dataset...")
        urllib.request.urlretrieve(url, zip_path)

        # Extract the dataset
        print("Extracting ShapeNetV2 dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)

        # Remove the zip file
        os.remove(zip_path)
        print("ShapeNetV2 dataset downloaded and extracted successfully.")

def get_dataset(dataroot, npoints, category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)
    te_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std,
    )
    return tr_dataset, te_dataset


# def get_dataloader(opt=Non, train_dataset, test_dataset=None):

#     if opt.distribution_type == 'multi':
#         train_sampler = torch.utils.data.distributed.DistributedSampler(
#             train_dataset,
#             num_replicas=opt.world_size,
#             rank=opt.rank
#         )
#         if test_dataset is not None:
#             test_sampler = torch.utils.data.distributed.DistributedSampler(
#                 test_dataset,
#                 num_replicas=opt.world_size,
#                 rank=opt.rank
#             )
#         else:
#             test_sampler = None
#     else:
#         train_sampler = None
#         test_sampler = None

#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
#                                                    shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)

#     if test_dataset is not None:
#         test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=test_sampler,
#                                                    shuffle=False, num_workers=int(opt.workers), drop_last=False)
#     else:
#         test_dataloader = None

#     return train_dataloader, test_dataloader, train_sampler, test_sampler
