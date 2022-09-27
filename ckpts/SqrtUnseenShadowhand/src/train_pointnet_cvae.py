import os
import argparse
import time

from utils_data.CMapDataset import CMapDataset
from models.PointNetCVAE import PointNetCVAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.set_seed import set_global_seed
from criterion import VAECriterion, VAEAttnCriterion
from plotly import graph_objects as go
from utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh

import trimesh as tm
from tqdm import tqdm
import shutil
import numpy as np
import torch
import torch.nn as nn
import sys
import platform
import torch.optim as optim
import random
import json
import math


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', default='debug', type=str)
    parser.add_argument('--id', default=0, type=int)

    parser.add_argument('--batchsize', default=4, type=int)
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lw_recon', default=100., type=float)  # sqrt(MSE(x, y))
    parser.add_argument('--lw_kld', default=1., type=float)
    parser.add_argument('--ann_temp', default=1., type=float)
    parser.add_argument('--ann_per_epochs', default=2, type=int)

    parser.add_argument('--disable_shadowhand', default=False, action='store_true')
    parser.add_argument('--disable_allegro', default=False, action='store_true')
    parser.add_argument('--disable_robotiq_3finger', default=False, action='store_true')
    parser.add_argument('--disable_barrett', default=False, action='store_true')
    parser.add_argument('--disable_ezgripper', default=False, action='store_true')

    parser.add_argument('--disable_attn_loss', default=False, action='store_true')
    parser.add_argument('--attn_loss_alpha', default=3., type=float)

    parser.add_argument('--batches_per_print', default=500, type=int)
    parser.add_argument('--seed', default=42, type=int)

    # parser.add_argument('--enable_only_barrett', default=False, action='store_true')
    args = parser.parse_args()
    time_tag = str(time.time())
    return args, time_tag


def plot_mesh_from_name(dataset_object_name):
    dataset_name = dataset_object_name.split('+')[0]
    object_name = dataset_object_name.split('+')[1]
    mesh_path = os.path.join('data/object', dataset_name, object_name, f'{object_name}.stl')
    object_mesh = tm.load(mesh_path)
    return plot_mesh(object_mesh, color='lightblue')


def visualize_results(object_list, object_point_clouds, domain: str, num_per_object=2):
    global vis_dir, i_epoch
    with torch.no_grad():
        model.eval()
        vis_bs = len(object_list)
        for i_iter in range(num_per_object):
            z_latent_code = torch.randn(vis_bs, model.latent_size, device=device).float()
            cmap_values = model.inference(object_point_clouds, z_latent_code)
            for i_vis in range(vis_bs):
                vis_data = [plot_point_cloud_cmap(object_point_clouds[i_vis, :, :3].cpu().detach().numpy(),
                                                  cmap_values[i_vis, :].cpu().detach().numpy())]
                vis_data += [plot_mesh_from_name(object_list[i_vis])]
                fig = go.Figure(data=vis_data)
                fig.write_html(os.path.join(vis_dir, f'epoch{i_epoch}-{domain}-{object_list[i_vis]}-{i_iter}.html'))


def validate(args,
             dataloader: DataLoader):
    global model, criterion, writer, i_epoch, weight_dir, best_epoch_record
    with torch.no_grad():
        model.eval()
        num_batches = len(dataloader)

        loss_history = []
        loss_kld_history = []
        loss_recon_history = []
        for data in tqdm(dataloader, desc=f'EPOCH[{i_epoch}/{args.n_epochs}]'):
            cmap, robot_name, object_name = data
            cmap_values_gt = cmap[:, :, 3]
            cmap_values_hat, means, logvars, z_latent_code = model(cmap)
            loss, loss_recon, loss_kld = criterion(means, logvars, cmap_values_gt, cmap_values_hat)

            loss = loss.item()
            loss_recon = loss_recon.item()
            loss_kld = loss_kld.item()
            loss_history.append(loss)
            loss_kld_history.append(loss_kld)
            loss_recon_history.append(loss_recon)

        loss = np.mean(loss_history)
        loss_kld = np.mean(loss_kld_history)
        loss_recon = np.mean(loss_recon_history)
        writer.add_scalar('validate/loss/loss', loss, global_step=i_epoch)
        writer.add_scalar('validate/loss/loss_recon', loss_recon, global_step=i_epoch)
        writer.add_scalar('validate/loss/loss_kld', loss_kld, global_step=i_epoch)
        print(f'[validate] loss: {loss}\n'
              f'           loss_recon: {loss_recon}\n'
              f'           loss_kld: {loss_kld}\n')
        if loss_recon < best_epoch_record['loss_recon_val']:
            print('update record and save model...')
            best_epoch_record['i_epoch'] = i_epoch
            best_epoch_record['loss_recon_val'] = loss_recon
            best_epoch_record['loss_kld_val'] = loss_kld
            json.dump(best_epoch_record, open(os.path.join(weight_dir, 'best_epoch_record.json'), 'w'))
            torch.save(model.state_dict(), os.path.join(weight_dir, 'pointnet_cvae_model.pth'))


def train(args,
          dataloader: DataLoader):
    global model, optimizer, criterion, writer, i_epoch
    model.train()
    num_batches = len(dataloader)

    loss_history = []
    loss_kld_history = []
    loss_recon_history = []
    i_batch = 0
    for data in tqdm(dataloader, desc=f'EPOCH[{i_epoch}/{args.n_epochs}]'):
        cmap, robot_name, object_name = data
        cmap_values_gt = cmap[:, :, 3]
        cmap_values_hat, means, logvars, z_latent_code = model(cmap)
        loss, loss_recon, loss_kld = criterion(means, logvars, cmap_values_gt, cmap_values_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        loss_recon = loss_recon.item()
        loss_kld = loss_kld.item()
        loss_history.append(loss)
        loss_kld_history.append(loss_kld)
        loss_recon_history.append(loss_recon)
        if i_batch % args.batches_per_print == args.batches_per_print - 1:
            step = i_epoch * math.floor(num_batches / args.batches_per_print) + math.floor(i_batch / args.batches_per_print)
            loss = np.mean(loss_history)
            loss_kld = np.mean(loss_kld_history)
            loss_recon = np.mean(loss_recon_history)
            writer.add_scalar('train/criterion/lw_recon', criterion.lw_recon, global_step=step)
            writer.add_scalar('train/criterion/lw_kld', criterion.lw_kld, global_step=step)

            writer.add_scalar('train/loss/loss', loss, global_step=step)
            writer.add_scalar('train/loss/loss_recon', loss_recon, global_step=step)
            writer.add_scalar('train/loss/loss_kld', loss_kld, global_step=step)
            print(f'[{i_batch}/{num_batches}] loss: {loss}\n'
                  f'                          loss_recon: {loss_recon}\n'
                  f'                          loss_kld: {loss_kld}\n')

        i_batch += 1
    criterion.apply_iter()


if __name__ == '__main__':
    start_time = time.time()
    args, time_tag = get_parser()
    set_global_seed(seed=args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('ARGUMENTS')
    print(args)

    # disable unseen robotic hand
    robot_name_list = ['ezgripper', 'barrett', 'robotiq_3finger', 'allegro', 'shadowhand']
    if args.disable_shadowhand:
        robot_name_list.remove('shadowhand')
    if args.disable_allegro:
        robot_name_list.remove('allegro')
    if args.disable_robotiq_3finger:
        robot_name_list.remove('robotiq_3finger')
    if args.disable_barrett:
        robot_name_list.remove('barrett')
    if args.disable_ezgripper:
        robot_name_list.remove('ezgripper')
    print(f'robot name list: {robot_name_list}')

    if args.disable_attn_loss:
        assert (abs(args.attn_loss_alpha - 1) < 1e-5)

    # 0. prepare logs basedir
    log_dir = os.path.join('logs', 'PointNet-CVAE', f'exp-{args.id}-{args.comment}_{time_tag}')
    if not args.disable_attn_loss:
        log_dir = os.path.join('logs', 'AttnCriterion', 'PointNet-CVAE', f'exp-{args.id}-{args.comment}_{time_tag}')
    weight_dir = os.path.join(log_dir, 'weights')
    tb_dir = os.path.join(log_dir, 'tb_dir')
    vis_dir = os.path.join(log_dir, 'vis_dir')
    shutil.rmtree(log_dir, ignore_errors=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'src'), exist_ok=True)
    for fn in os.listdir('.'):
        if fn[-3:] == '.py':
            fn = os.path.join(fn)
            shutil.copy(fn, os.path.join(log_dir, 'src', fn))
    for fn in os.listdir('./models'):
        if fn[-3:] == '.py':
            fn = os.path.join('models', fn)
            os.makedirs(os.path.join(log_dir, 'src', 'models'), exist_ok=True)
            shutil.copy(fn, os.path.join(log_dir, 'src', fn))
    for fn in os.listdir('./utils'):
        if fn[-3:] == '.py':
            fn = os.path.join('utils', fn)
            os.makedirs(os.path.join(log_dir, 'src', 'utils'), exist_ok=True)
            shutil.copy(fn, os.path.join(log_dir, 'src', fn))
    for fn in os.listdir('./utils_data'):
        if fn[-3:] == '.py':
            fn = os.path.join('utils_data', fn)
            os.makedirs(os.path.join(log_dir, 'src', 'utils_data'), exist_ok=True)
            shutil.copy(fn, os.path.join(log_dir, 'src', fn))
    f = open(os.path.join(log_dir, 'command.txt'), 'w')
    f.write(' '.join(sys.argv))
    f.close()
    writer = SummaryWriter(log_dir=tb_dir)

    # 1. prepare dataset dir
    if platform.node() == 'LAPTOP-F8TAF641':
        raise NotImplementedError
    elif platform.node() == 'puhao-Nvidia-3090Ti':
        dataset_basedir = '/home/puhao/data/CMap-Dataset/ContactMapDataset_align'
    elif platform.node()[:6] == 'lambda':
        dataset_basedir = '/home/lipuhao/data/ContactMapDataset_align'
    else:
        raise NotImplementedError()

    # todo: to test a sqrt aligned dist map
    if platform.node() == 'LAPTOP-F8TAF641':
        raise NotImplementedError
    elif platform.node() == 'puhao-Nvidia-3090Ti':
        dataset_basedir = '/home/puhao/data/CMap-Dataset-sqrt/ContactMapDataset_sqrt_align'
    elif platform.node()[:6] == 'lambda':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    # 2. load dataset and build dataloader
    print(f'training on: {platform.node()}')
    print(f'initialize CMap Dataset from: {dataset_basedir}')
    batchsize = args.batchsize
    train_dataset = CMapDataset(dataset_basedir=dataset_basedir, mode='train', device=device,
                                robot_name_list=robot_name_list)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    validate_dataset = CMapDataset(dataset_basedir=dataset_basedir, mode='validate', device=device)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    train_object_list = train_dataset.object_list
    validate_object_list = validate_dataset.object_list
    object_pcs = train_dataset.object_point_clouds

    train_object_pcs = torch.stack([object_pcs[x][:, :3] for x in train_object_list], dim=0).to(device)
    validate_object_pcs = torch.stack([object_pcs[x][:, :3] for x in validate_object_list], dim=0).to(device)
    print('finish CMap Dataset init...')

    # 3. init model
    print('init PointNet-CVAE model from scratch...')
    model = PointNetCVAE(latent_size=128,
                         encoder_layers_size=[4, 64, 128, 512],
                         decoder_global_feat_size=512,
                         decoder_pointwise_layers_size=[3, 64, 64],
                         decoder_global_layers_size=[64, 128, 512],
                         decoder_decoder_layers_size=[64+512+128, 512, 64, 64, 1])
    model = model.to(device)
    print('finish init model...')

    # 4. init optimizer, criterion, metrics
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if args.disable_attn_loss:
        criterion = VAECriterion(lw_init_recon=args.lw_recon, lw_init_kld=args.lw_kld,
                                 ann_temp=args.ann_temp, ann_per_epochs=args.ann_per_epochs,
                                 batchsize=args.batchsize)
    else:
        criterion = VAEAttnCriterion(lw_init_recon=args.lw_recon, lw_init_kld=args.lw_kld,
                                     ann_temp=args.ann_temp, ann_per_epochs=args.ann_per_epochs,
                                     batchsize=args.batchsize, alpha=args.attn_loss_alpha)

    # 5. start training
    best_epoch_record = {'i_epoch': 0,
                         'loss_kld_val': 10000,
                         'loss_recon_val': 10000}
    print(f'n_epochs: {args.n_epochs} | training...')
    json.dump(best_epoch_record, open(os.path.join(weight_dir, 'best_epoch_record.json'), 'w'))
    torch.save(model.state_dict(), os.path.join(weight_dir, 'pointnet_cvae_model.pth'))
    for i_epoch in range(args.n_epochs):
        train(args, train_dataloader)
        validate(args, validate_dataloader)
        # if i_epoch % args.ann_per_epochs == args.ann_per_epochs - 1:
        #     visualize_results(validate_object_list, validate_object_pcs, 'validate', 4)
        #     visualize_results(train_object_list, train_object_pcs, 'train', 1)
    print('finish training...')
    writer.close()
    print(f'consuming time: {time.time() - start_time}')
