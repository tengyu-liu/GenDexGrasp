import argparse
import json
import os.path
import time
import sys
import shutil

import trimesh.sample

import torch
import plotly.graph_objects as go
from utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh_from_name
from utils.set_seed import set_global_seed
from torch.utils.tensorboard import SummaryWriter
import trimesh as tm
import torch.nn as nn


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', default='sharp_clamp', type=str)

    parser.add_argument('--s_model', default='PointNetCVAE_UnseenShadowhandA4', type=str)

    parser.add_argument('--num_per_seen_object', default=4, type=int)
    parser.add_argument('--num_per_unseen_object', default=16, type=int)

    parser.add_argument('--comment', default='debug', type=str)
    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


def pre_process_sharp_clamp(contact_map):
    gap_th = 0.5  # delta_th = (1 - gap_th)
    gap_th = min(contact_map.max().item(), gap_th)
    delta_th = (1 - gap_th)
    contact_map[contact_map > 0.4] += delta_th
    # contact_map += delta_th
    contact_map = torch.clamp_max(contact_map, 1.)
    return contact_map


def identity_map(contact_map):
    return contact_map


if __name__ == '__main__':
    set_global_seed(seed=42)
    args, time_tag = get_parser()

    pre_process_map = {'sharp_clamp': pre_process_sharp_clamp,
                       'identity': identity_map}
    pre_process_contact_map_goal = pre_process_map[args.pre_process]

    logs_basedir = os.path.join('logs_inf', f'{args.s_model}', f'{args.pre_process}', f'{args.comment}-{time_tag}')
    vis_id_dir = os.path.join(logs_basedir, 'vis_id_dir')
    vis_ood_dir = os.path.join(logs_basedir, 'vis_ood_dir')
    cmap_path_id = os.path.join(logs_basedir, 'cmap_id.pt')
    cmap_path_ood = os.path.join(logs_basedir, 'cmap_ood.pt')
    os.makedirs(logs_basedir, exist_ok=False)
    os.makedirs(vis_id_dir, exist_ok=False)
    os.makedirs(vis_ood_dir, exist_ok=False)

    device = "cuda"
    if args.s_model == 'PointNetCVAE_UnseenShadowhandA3':
        model_basedir = 'models/UnseenShadowhandA3'
        from models.UnseenShadowhandA3.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_UnseenShadowhandA4':
        model_basedir = 'models/UnseenShadowhandA4'
        from models.UnseenShadowhandA3.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_OnlyBarrettA4':
        model_basedir = 'models/OnlyBarrettA4'
        from models.OnlyBarrettA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_UnseenBarrettA4':
        model_basedir = 'models/UnseenBarrettA4'
        from models.UnseenBarrettA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_UnseenEzgripperA4':
        model_basedir = 'models/UnseenEzgripperA4'
        from models.UnseenEzgripperA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_FullRobotsA4':
        model_basedir = 'models/FullRobotsA4'
        from models.FullRobotsA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_UnseenAllegroA4':
        model_basedir = 'models/UnseenAllegroA4'
        from models.UnseenAllegroA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_UnseenRobotiq3FA4':
        model_basedir = 'models/UnseenRobotiq3FA4'
        from models.UnseenRobotiq3FA4.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']

    elif args.s_model == 'PointNetCVAE_SqrtUnseenShadowhand':
        model_basedir = 'models/SqrtUnseenShadowhand'
        from models.SqrtUnseenShadowhand.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_SqrtUnseenBarrett':
        model_basedir = 'models/SqrtUnseenBarrett'
        from models.SqrtUnseenBarrett.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    elif args.s_model == 'PointNetCVAE_SqrtUnseenEzgripper':
        model_basedir = 'models/SqrtUnseenEzgripper'
        from models.SqrtUnseenEzgripper.src.models.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    else:
        raise NotImplementedError("Occur when load model...")

    cmap_ood = []
    for object_name in unseen_object_list:
        print(f'unseen object name: {object_name}')
        object_mesh: tm.Trimesh
        object_mesh = tm.load(os.path.join('data/object', object_name.split('+')[0], object_name.split("+")[1],
                                           f'{object_name.split("+")[1]}.stl'))
        for i_sample in range(args.num_per_unseen_object):
            cmap_ood_sample = {'object_name': object_name,
                              'i_sample': i_sample,
                              'object_point_cloud': None,
                              'contact_map_value': None}
            print(f'[{i_sample}/{args.num_per_unseen_object}] | {object_name}')
            object_point_cloud, faces_indices = trimesh.sample.sample_surface(mesh=object_mesh, count=2048)
            contact_points_normal = torch.tensor([object_mesh.face_normals[x] for x in faces_indices]).float()
            object_point_cloud = torch.Tensor(object_point_cloud).float()
            object_point_cloud = torch.cat([object_point_cloud, contact_points_normal], dim=1).to(device)
            z_latent_code = torch.randn(1, model.latent_size, device=device).float()
            contact_map_value = model.inference(object_point_cloud[:, :3].unsqueeze(0), z_latent_code).squeeze(0)
            # process the contact map value
            contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
            contact_map_value = pre_process_contact_map_goal(contact_map_value).to(device)
            contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1)

            cmap_ood_sample['object_point_cloud'] = object_point_cloud
            cmap_ood_sample['contact_map_value'] = contact_map_value
            cmap_ood.append(cmap_ood_sample)
            vis_data = []
            vis_data += [plot_point_cloud_cmap(contact_map_goal[:, :3].cpu().detach().numpy(),
                                               contact_map_goal[:, 6].cpu().detach().numpy())]
            vis_data += [plot_mesh_from_name(f'{object_name}')]
            fig = go.Figure(data=vis_data)
            fig.write_html(os.path.join(vis_ood_dir, f'unseen-{object_name}-{i_sample}.html'))
    torch.save(cmap_ood, cmap_path_ood)

    cmap_id = []
    for object_name in seen_object_list:
        print(f'seen object name: {object_name}')
        object_mesh: tm.Trimesh
        object_mesh = tm.load(os.path.join('data/object', object_name.split('+')[0], object_name.split("+")[1],
                                           f'{object_name.split("+")[1]}.stl'))
        for i_sample in range(args.num_per_seen_object):
            cmap_id_sample = {'object_name': object_name,
                               'i_sample': i_sample,
                               'object_point_cloud': None,
                               'contact_map_value': None}
            print(f'[{i_sample}/{args.num_per_seen_object}] | {object_name}')
            object_point_cloud, faces_indices = trimesh.sample.sample_surface(mesh=object_mesh, count=2048)
            contact_points_normal = torch.tensor([object_mesh.face_normals[x] for x in faces_indices]).float()
            object_point_cloud = torch.Tensor(object_point_cloud).float()
            object_point_cloud = torch.cat([object_point_cloud, contact_points_normal], dim=1).to(device)
            z_latent_code = torch.randn(1, model.latent_size, device=device).float()
            contact_map_value = model.inference(object_point_cloud[:, :3].unsqueeze(0), z_latent_code).squeeze(0)
            # process the contact map value
            contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
            contact_map_value = pre_process_contact_map_goal(contact_map_value).to(device)
            contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1)

            cmap_id_sample['object_point_cloud'] = object_point_cloud
            cmap_id_sample['contact_map_value'] = contact_map_value
            cmap_id.append(cmap_id_sample)
            vis_data = []
            vis_data += [plot_point_cloud_cmap(contact_map_goal[:, :3].cpu().detach().numpy(),
                                               contact_map_goal[:, 6].cpu().detach().numpy())]
            vis_data += [plot_mesh_from_name(f'{object_name}')]
            fig = go.Figure(data=vis_data)
            fig.write_html(os.path.join(vis_id_dir, f'seen-{object_name}-{i_sample}.html'))
    torch.save(cmap_id, cmap_path_id)
