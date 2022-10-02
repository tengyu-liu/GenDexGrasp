'''
LastEditTime: 2022-05-23 19:24:38
Description: Your description
Date: 2021-11-04 04:54:29
Author: Aiden Li
LastEditors: Aiden Li (i@aidenli.net)
'''
import io
import os
from tkinter.messagebox import NO
import numpy as np
import torch
import torch
import trimesh as tm
from plotly import graph_objects as go
from PIL import Image

colors = [
    'blue', 'red', 'yellow', 'pink', 'gray', 'orange'
]

def plot_mesh(mesh, color='lightblue', opacity=1.0):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity)

def plot_hand(verts, faces, color='lightpink', opacity=1.0):
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color, opacity=opacity)

def plot_contact_points(pts, grad, color='lightpink'):
    pts = pts.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return go.Cone(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], u=-grad[:, 0], v=-grad[:, 1], w=-grad[:, 2], anchor='tip',
                   colorscale=[(0, color), (1, color)], sizemode='absolute', sizeref=0.2, opacity=0.5)

def plot_point_cloud(pts, color='lightblue', mode='markers'):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(
            color=color,
            size=3.
        )
    )


occ_cmap = lambda levels, thres=0.: [f"rgb({int(255)},{int(255)},{int(255)})" if x > thres else
                           f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_point_cloud_occ(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': occ_cmap(color_levels),
            'size': 3,
            'opacity': 1
        }
    )


contact_cmap = lambda levels, thres=0.: [f"rgb({int(255 * (1 - x))},{int(255 * (1 - x))},{int(255 * (1 - x))})" if x >= thres else
                                         f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]

def plot_point_cloud_cmap(pts, color_levels=None):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': contact_cmap(color_levels),
            'size': 3.5,
            'opacity': 1
        }
    )


normal_color_map = lambda levels, thres=0., color_scale=8.: [f"rgb({int(255 * (color_scale * x[0]))},{int(255 * (color_scale * x[1]))},{int(255 * (color_scale * x[2]))})" if x[0] >= thres else
                                                             f"rgb({int(0)},{int(0)},{int(0)})" for x in levels.tolist()]


def plot_normal_map(pts, normal):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={
            'color': normal_color_map(np.abs(normal)),
            'size': 3.5,
            'opacity': 1
        }
    )


def plot_grasps(directory, tag, uuids, physics_guide, handcodes, contact_idx, ret_plots=False, save_html=True, include_contacts=True):
    handcode = handcodes[:, -1]
    hand_vertices = physics_guide.get_vertices(handcodes)
    hand_faces = physics_guide.hand_model.faces
    
    object_models = physics_guide.object_models
        
    if include_contacts:
        contact_points = []
        for ind in range(contact_idx.shape[1]):
            contact_point_vertices = torch.gather(
                hand_vertices, 1,
                contact_idx[:, ind].unsqueeze(-1).tile((1, 1, 3))
            )
            contact_points.append(contact_point_vertices.detach().cpu().numpy())
            
    hand_vertices = hand_vertices.detach().cpu().numpy()
    
    plots = []

    for batch_idx in range(hand_vertices.shape[0]):
        to_plot = []

        to_plot.append(plot_hand(hand_vertices[batch_idx], hand_faces))

        for obj_ind, obj in enumerate(object_models):
            to_plot.append(obj.get_plot(batch_idx))
            if include_contacts:
                to_plot.append(plot_point_cloud(contact_points[obj_ind][batch_idx], color=colors[obj_ind]))
        
        fig = go.Figure(to_plot)
        
        if save_html:
            fig.write_html(os.path.join(f"{ directory }", f"fig-{ str(uuids[ batch_idx ]) }-{ batch_idx }-{ tag }.html"))
        if ret_plots:
            plots.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(fig.to_image(format="png", width=1280, height=720))))))
            
    if ret_plots:
        return plots
    

def plot_mesh_from_name(dataset_object_name):
    dataset_name = dataset_object_name.split('+')[0]
    object_name = dataset_object_name.split('+')[1]
    mesh_path = os.path.join('data/object', dataset_name, object_name, f'{object_name}.stl')
    object_mesh = tm.load(mesh_path)
    return plot_mesh(object_mesh, color='lightblue')

