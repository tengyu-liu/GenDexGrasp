"""
    Implementation of Adam of Contact Map Guide Energy
"""
import collections

from scipy.spatial.transform import Rotation as R
import torch
from utils.get_models import get_handmodel
import torch.nn.functional as F


class CMapAdam:
    def __init__(self, robot_name, contact_map_goal=None,
                 num_particles=32, init_rand_scale=0.5,
                 learning_rate=5e-3, running_name=None, energy_func_name='align_dist',
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose_energy=False):
        self.running_name = running_name
        self.device = device
        self.robot_name = robot_name
        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.verbose_energy = verbose_energy

        self.global_step = None
        self.contact_map_goal = None
        self.q_current = None
        self.energy = None

        self.compute_energy = None
        self.object_radius = None
        self.contact_value_goal = None
        self.object_point_cloud = None
        self.object_normal_cloud = None

        self.q_global = None
        self.q_local = None
        self.optimizer = None

        self.handmodel = get_handmodel(robot_name, num_particles, device, hand_scale=1.)
        # self.handmodel = get_handmodel(robot_name, 1, device, hand_scale=1.)
        self.q_joint_lower = self.handmodel.revolute_joints_q_lower.detach()
        self.q_joint_upper = self.handmodel.revolute_joints_q_upper.detach()
        # self.q_joint_lower.requires_grad = True
        # self.q_joint_upper.requires_grad = True

        if contact_map_goal is not None:
            self.reset(contact_map_goal=contact_map_goal, running_name=running_name, energy_func_name=energy_func_name)

    def reset(self, contact_map_goal, running_name, energy_func_name):
        self.handmodel = get_handmodel(self.robot_name, self.num_particles, self.device, hand_scale=1.)
        energy_func_map = {'euclidean_dist': self.compute_energy_euclidean_dist,
                           'align_dist': self.compute_energy_align_dist}
        self.compute_energy = energy_func_map[energy_func_name]

        self.running_name = running_name
        self.is_pruned = False
        self.best_index = None
        self.global_step = 0
        self.distance_init = 1.
        self.contact_map_goal = contact_map_goal.to(self.device)
        self.object_point_cloud = contact_map_goal[:, :3].to(self.device)
        self.object_normal_cloud = contact_map_goal[:, 3:6].to(self.device)
        self.contact_value_goal = contact_map_goal[:, 6].to(self.device)
        self.object_radius = torch.max(torch.norm(self.object_point_cloud, dim=1, p=2))
        self.q_current = torch.zeros(self.num_particles, 3 + 6 + len(self.handmodel.revolute_joints),
                                     device=self.device)
        random_rot = torch.tensor(R.random(self.num_particles).as_dcm(), device=self.device).float()

        self.q_current[:, 3:9] = random_rot.reshape(self.num_particles, 9)[:, :6]
        # # TODO: for debug
        # self.handmodel.update_kinematics(q=self.q_current)
        hand_center_position = torch.mean(self.handmodel.get_surface_points(q=self.q_current), dim=1)
        if self.robot_name == 'allegro' or self.robot_name == 'robotiq_3finger_real_robot':
            hand_center_position = torch.mean(self.handmodel.get_surface_points_paml(q=self.q_current), dim=1)
        if self.robot_name == 'barrett':
            hand_normal = torch.Tensor([[0., 0., 1.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'allegro_old':
            hand_normal = torch.Tensor([[1., 0., 0.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'shadowhand':
            hand_normal = torch.Tensor([[0., -1., 0.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'robotiq_3finger':
            hand_normal = 1.3 * torch.Tensor([[0., 0., 1.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'robotiq_3finger_real_robot':
            hand_normal = 2. * torch.Tensor([[0., 1., 0.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'ezgripper':
            hand_normal = torch.Tensor([[1.3, 0., 0.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        elif self.robot_name == 'allegro':
            hand_normal = 1.2 * torch.Tensor([[1., 0., 0.]]).to(self.device).T.float()
            hand_normal *= self.distance_init
            hand_normal = torch.einsum('bmn,nk->bmk', random_rot.transpose(2, 1), hand_normal).squeeze(2)
        else:
            raise NotImplementedError()

        self.q_current[:, :3] = -hand_center_position
        self.q_current[:, :3] -= hand_normal * self.object_radius
        self.q_current[:, 9:] = self.init_random_scale * torch.rand_like(self.q_current[:, 9:]) * (self.q_joint_upper - self.q_joint_lower) + self.q_joint_lower
        # self.q_current[:, 9:] = torch.zeros_like(self.q_current[:, 9:])

        # self.q_current = self.q_current.detach()
        self.q_current.requires_grad = True
        # self.optimizer_global = torch.optim.Adam([self.q_global], lr=self.learning_rate)
        # self.optimizer_local = torch.optim.Adam([self.q_local], lr=self.learning_rate)
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def compute_energy_euclidean_dist(self):
        hand_surface_points_ = self.handmodel.get_surface_points()
        hand_surface_points = hand_surface_points_.clone()
        # compute contact value with align dist
        npts_object = self.object_point_cloud.size()[0]
        npts_hand = hand_surface_points.size()[1]
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
        batch_object_point_cloud = batch_object_point_cloud.reshape(self.num_particles, 1, npts_object, 3)
        hand_surface_points = hand_surface_points.reshape(self.num_particles, 1, npts_hand, 3)
        batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1).transpose(1, 2)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)

        contact_dist = object_hand_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (torch.sigmoid(100 * contact_dist) - 0.5)
        energy_contact = torch.abs(contact_value_current - self.contact_value_goal.reshape(1, -1)).mean(dim=1)

        # compute penetration
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
        batch_object_point_cloud = batch_object_point_cloud.reshape(self.num_particles, 1, npts_object, 3)
        hand_surface_points = hand_surface_points_.reshape(self.num_particles, 1, npts_hand, 3)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1).transpose(1, 2)
        batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1)
        hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack([self.object_point_cloud[x, :] for x in hand_object_indices], dim=0)
        hand_object_normal = torch.stack([self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0)
        hand_object_signs = ((hand_object_points - hand_surface_points_) * hand_object_normal).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)
        energy = energy_contact + 100 * energy_penetration
        self.energy = energy
        # TODO: add a normalized energy
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(self.q_joint_lower - self.q_current[:, 9:])
        self.energy = energy + z_norm.sum(dim=1)
        if self.verbose_energy:
            return energy, energy_penetration, z_norm
        else:
            return energy

    def compute_energy_align_dist(self):
        # hand_surface_points_ = self.handmodel.get_surface_points()
        hand_surface_points_ = self.handmodel.get_surface_points_new()
        hand_surface_points = hand_surface_points_.clone()
        # compute contact value with align dist
        npts_object = self.object_point_cloud.size()[0]
        npts_hand = hand_surface_points.size()[1]
        with torch.no_grad():
            batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
            batch_object_point_cloud = batch_object_point_cloud.view(self.num_particles, 1, npts_object, 3)
            batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1).transpose(1, 2)
        hand_surface_points = hand_surface_points.view(self.num_particles, 1, npts_hand, 3)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        with torch.no_grad():
            batch_object_normal_cloud = self.object_normal_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
            batch_object_normal_cloud = batch_object_normal_cloud.view(self.num_particles, 1, npts_object, 3)
            batch_object_normal_cloud = batch_object_normal_cloud.repeat(1, npts_hand, 1, 1).transpose(1, 2)
        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        object_hand_align = ((hand_surface_points - batch_object_point_cloud) *
                             batch_object_normal_cloud).sum(dim=3)
        object_hand_align /= (object_hand_dist + 1e-5)

        object_hand_align_dist = object_hand_dist * torch.exp(2 * (1 - object_hand_align))
        # TODO: add a mask of back points
        # object_hand_align_dist = torch.where(object_hand_align > 0, object_hand_align_dist,
        #                                      torch.ones_like(object_hand_align_dist))

        contact_dist = torch.sqrt(object_hand_align_dist.min(dim=2)[0])
        # contact_dist = object_hand_align_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)
        energy_contact = torch.abs(contact_value_current - self.contact_value_goal.view(1, -1)).mean(dim=1)

        # compute penetration
        with torch.no_grad():
            batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
            batch_object_point_cloud = batch_object_point_cloud.view(self.num_particles, 1, npts_object, 3)
            batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1)

        hand_surface_points = hand_surface_points_.view(self.num_particles, 1, npts_hand, 3)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1).transpose(1, 2)
        hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack([self.object_point_cloud[x, :] for x in hand_object_indices], dim=0)
        hand_object_normal = torch.stack([self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0)
        # torch.gather()
        hand_object_signs = ((hand_object_points - hand_surface_points_) * hand_object_normal).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)

        energy = energy_contact + 100 * energy_penetration
        # energy = energy_contact
        # TODO: add a normalized energy
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(self.q_joint_lower - self.q_current[:, 9:])
        if self.robot_name == 'robotiq_3finger':
            self.energy = energy
        elif self.robot_name == 'robotiq_3finger_real_robot':
            # z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(self.q_joint_lower - self.q_current[:, 9:])
            q_joint_mid = (self.q_joint_lower + self.q_joint_upper) / 2
            q_joint_mid = (self.q_joint_lower + q_joint_mid) / 2
            z_norm = torch.abs(self.q_current[:, 9:] - q_joint_mid).sum(dim=1)
            self.energy = energy + z_norm * 0.2
        else:
            self.energy = energy + z_norm.sum(dim=1)

        if self.verbose_energy:
            return energy, energy_penetration, z_norm
        else:
            return energy

    def step(self):
        self.optimizer.zero_grad()
        self.handmodel.update_kinematics(q=self.q_current)
        energy = self.compute_energy()
        # if self.is_pruned:
        #     energy = energy[self.best_index]
        energy.mean().backward()
        self.optimizer.step()
        self.global_step += 1

    def do_pruning(self):
        raise NotImplementedError
        self.best_index = self.energy.min(dim=0)[1].item()
        # todo: restart optimizer?
        self.handmodel = get_handmodel(self.robot_name, 1, self.device, hand_scale=1.)
        self.q_current = self.q_current[[self.best_index], :].detach()
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate / 5)
        self.is_pruned = True

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color='pink', opacity=0.7):
        # self.handmodel.update_kinematics(q=self.q_current)
        return self.handmodel.get_plotly_data(q=self.q_current, i=index, color=color, opacity=opacity)


if __name__ == '__main__':
    import argparse
    from utils.set_seed import set_global_seed
    import plotly.graph_objects as go
    from utils.visualize_plotly import plot_mesh_from_name

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_idx', default=100, type=int)
    parser.add_argument('--robot_name', default='robotiq_3finger_real_robot', type=str)
    args = parser.parse_args()
    bs = 1
    set_global_seed(43)
    robot_name = args.robot_name
    i_obj = args.obj_idx
    device = 'cuda'
    # contact_map_goal = torch.randn(2048, 4, device=device) * 0.05
    cmap_metadata = torch.load('dataset/UnseenShadowhand/SharpClamp_A4/cmap_ood.pt')[i_obj]
    object_name = cmap_metadata['object_name']
    object_point_cloud = cmap_metadata['object_point_cloud']
    contact_map_value = cmap_metadata['contact_map_value']

    opt_model = CMapAdam(robot_name=robot_name, contact_map_goal=None, num_particles=bs,
                         init_rand_scale=0.5, learning_rate=1e-4, energy_func_name='align_dist',
                         device=device)
    contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1).to(device)
    opt_model.reset(contact_map_goal, 'init', 'align_dist')
    init_opt_q = opt_model.get_opt_q()

    handmodel = get_handmodel(args.robot_name, bs, 'cuda', 1.)
    handmodel.update_kinematics(q=init_opt_q.clone())
    vis_data = []
    for i in range(init_opt_q.shape[0]):
        vis_data += handmodel.get_plotly_data(i=i, opacity=0.5, color='pink')
    vis_data += handmodel.get_plotly_data(i=0, opacity=0.5, color='pink')
    vis_data += [plot_mesh_from_name(object_name)]
    fig = go.Figure(data=vis_data)
    fig.show()
    pass