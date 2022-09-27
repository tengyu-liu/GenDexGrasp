"""
    Statistics for contact area
    down sample from CMap-Dataset-V0.0
"""
import torch
import os
import platform
from utils_data.CMapDataset import CMapDataset
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm, trange


if __name__ == '__main__':
    batchsize = 128
    num_down_sample_grasp = 1280
    log_dir = os.path.join('logs', 'Contact-Statistics', f'down-sample-datas')

    # 1. prepare dataset dir
    if platform.node() == 'LAPTOP-F8TAF641':
        dataset_basedir = "D:/BIGAI/Project/Data/Gen2Grasp-CMap-Dataset/CMap-Dataset-V0.0"
    elif platform.node() == 'puhao-Nvidia-3090Ti':
        dataset_basedir = "/home/puhao/data/CMap-Dataset/CMap-Dataset-V1.0"
    elif platform.node()[:6] == 'lambda':
        dataset_basedir = '/home/lipuhao/scratch/dev/gen2grasp/Data-Synthesis-yuyang/Data-Synthesis/CMap-Dataset-V0.0/'
    else:
        raise NotImplementedError()

    device = 'cuda'
    dataset = CMapDataset(dataset_basedir=dataset_basedir, mode='full', device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    num_batches = num_down_sample_grasp // batchsize
    cmap_value = []
    counter = 0
    for data in dataloader:
        if counter > (num_down_sample_grasp // num_batches):
            break
        cmap, _, _ = data
        cmap_value.append(cmap)
        counter += 1

    cmap_value = torch.cat(cmap_value, dim=0)
    cmap_value = cmap_value.view(-1).cpu().detach().numpy()

    plt.style.use('seaborn')
    sns.distplot(cmap_value, hist=False, kde=False, fit=stats.norm,
                 fit_kws={'color': 'red', 'label': 'Density of ContactValue', 'linestyle': '-'})

    # legend()显示图例，savefig()保存图片，show()绘图
    plt.legend()
    # plt.savefig('Demo.png')
    plt.show()


