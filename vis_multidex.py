import os
import argparse
import torch
import plotly.graph_objects as go
import trimesh as tm
from utils.get_models import get_handmodel
from utils.visualize_plotly import plot_mesh


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str)
    parser.add_argument('--object_name', default='contactdb+apple', type=str)
    parser.add_argument('--num_vis', default=8, type=int)
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = get_parser()
    robot_name = args.robot_name
    object_name = args.object_name
    num_vis = args.num_vis

    info = torch.load(f'MultiDex/{robot_name}/{robot_name}.pt')['info']
    print(f'data info: \n{info}')
    metadata = torch.load(f'MultiDex/{robot_name}/{robot_name}.pt')['metadata']
    metadata = [m for m in metadata if m[1] == object_name]
    hand_model = get_handmodel(robot_name, 1, 'cuda', 1.)
    for select_index in range(num_vis):
        q = metadata[select_index][0]
        object_name = metadata[select_index][1]
        vis_data = hand_model.get_plotly_data(q=q.cuda().unsqueeze(0), color='pink')
        object_mesh_basedir = 'data/object'
        object_mesh_path = os.path.join(object_mesh_basedir, f'{object_name.split("+")[0]}', f'{object_name.split("+")[1]}',
                                        f'{object_name.split("+")[1]}.stl')
        vis_data += [plot_mesh(mesh=tm.load(object_mesh_path))]
        fig = go.Figure(data=vis_data)
        fig.show()
