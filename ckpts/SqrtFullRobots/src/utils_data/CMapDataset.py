import os
import torch.utils.data as data
import torch
import pickle
import time
from tqdm import tqdm
import json
import numpy as np
from torch.utils.data import DataLoader
from plotly import graph_objects as go
from utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh
import trimesh as tm
from utils.set_seed import set_global_seed


class CMapDataset(data.Dataset):
    def __init__(self,
                 dataset_basedir='/home/puhao/data/CMap-Dataset/ContactMapDataset/',
                 object_npts=2048,
                 enable_disturb=True,
                 disturbance_sigma=0.001,  # meter
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 mode='train', robot_name_list=['ezgripper', 'barrett', 'robotiq_3finger', 'allegro', 'shadowhand']
                 ):
        self.device = device
        self.dataset_basedir = dataset_basedir
        self.object_npts = object_npts
        self.enable_disturb = enable_disturb
        self.disturbance_sigma = disturbance_sigma
        self.robot_name_list = robot_name_list

        print('loading cmap metadata....')
        cmap_dataset = torch.load(os.path.join(dataset_basedir, 'cmap_dataset.pt'))
        self.metadata_info = cmap_dataset['info']
        self.metadata = cmap_dataset['metadata']
        print('loading object point clouds....')
        self.object_point_clouds = torch.load(os.path.join(dataset_basedir, 'object_point_clouds.pt'))

        if mode == 'train':
            self.object_list = json.load(open(os.path.join(dataset_basedir, 'split_train_validate_objects.json'), 'rb'))[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == 'validate':
            self.object_list = json.load(open(os.path.join(dataset_basedir, 'split_train_validate_objects.json'), 'rb'))[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == 'full':
            self.object_list = json.load(open(os.path.join(dataset_basedir, 'split_train_validate_objects.json'), 'rb'))['train'] + json.load(open(os.path.join(dataset_basedir, 'split_train_validate_objects.json'), 'rb'))['validate']
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        else:
            raise NotImplementedError()
        print(f'object selection: {self.object_list}')

        self.datasize = len(self.metadata)
        print('finish loading dataset....')

    def __len__(self):
        return self.datasize

    def __getitem__(self, item):
        disturbance = torch.randn(self.object_npts, 3) * self.disturbance_sigma
        map_value = self.metadata[item][0]
        robot_name = self.metadata[item][3]
        object_name = self.metadata[item][2]
        contact_map = self.object_point_clouds[object_name] + disturbance * self.enable_disturb
        contact_map = torch.cat([contact_map, map_value], dim=1).to(self.device)

        return contact_map, robot_name, object_name


def plot_mesh_from_name(dataset_object_name):
    dataset_name = dataset_object_name.split('+')[0]
    object_name = dataset_object_name.split('+')[1]
    dataset_name_map = {'contactdb': 'ContactDB'}
    dataset_name = dataset_name_map[dataset_name]
    mesh_path = os.path.join('..', 'data', dataset_name, object_name, f'{object_name}_scaled.stl')
    object_mesh = tm.load(mesh_path)
    return plot_mesh(object_mesh, color='lightblue')


if __name__ == '__main__':
    set_global_seed(42)
    CMapDataset = CMapDataset(device='cuda', enable_disturb=False)
    dataloader = DataLoader(dataset=CMapDataset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=0)
    last_time_tag = time.time()
    for data in tqdm(dataloader):
        cmap, robot_name, object_name = data
        print(f'bs = {dataloader.batch_size} | time consume: {time.time() - last_time_tag}')
        # to show a cmap
        cmap = cmap[0]
        vis_data = [plot_point_cloud_cmap(cmap[:, :3].cpu().detach().numpy(), cmap[:, 3].cpu().detach().numpy())]
        # vis_data += [plot_mesh_from_name(object_name[0])]
        fig = go.Figure(data=vis_data)
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(visible=False),
        #         yaxis=dict(visible=False),
        #         zaxis=dict(visible=False)
        #     )
        # )
        fig.show()
        # vis_data = [plot_point_cloud_cmap(cmap[:, :3].cpu().detach().numpy(),
        #                                   torch.zeros_like(cmap[:, 3]).cpu().detach().numpy())]
        # vis_data += [plot_mesh_from_name(object_name[0])]
        # fig = go.Figure(data=vis_data)
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(visible=False),
        #         yaxis=dict(visible=False),
        #         zaxis=dict(visible=False)
        #     )
        # )
        # fig.show()
        # input('Enter here to continue: ')
        last_time_tag = time.time()

