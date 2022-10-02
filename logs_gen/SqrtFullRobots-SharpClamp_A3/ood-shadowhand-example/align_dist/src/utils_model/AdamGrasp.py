from utils_model.CMapAdam import CMapAdam
import torch
from tqdm import tqdm


class AdamGrasp:
    def __init__(self, robot_name, writer, contact_map_goal=None,
                 num_particles=32, init_rand_scale=0.5, max_iter=300, steps_per_iter=2,
                 learning_rate=5e-3, device='cuda', energy_func_name='align_dist'):
        self.writer = writer
        self.robot_name = robot_name
        self.contact_map_goal = contact_map_goal
        self.num_particles = num_particles
        self.init_rand_scale = init_rand_scale
        self.learning_rate = learning_rate
        self.device = device
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.energy_func_name = energy_func_name

        self.opt_model = CMapAdam(robot_name=robot_name, contact_map_goal=None, num_particles=self.num_particles,
                                  init_rand_scale=init_rand_scale, learning_rate=learning_rate, energy_func_name=self.energy_func_name,
                                  device=device)

    def run_adam(self, object_name, contact_map_goal, running_name):
        q_trajectory = []
        self.opt_model.reset(contact_map_goal, running_name, self.energy_func_name)
        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())
        iters_per_print = self.max_iter // 4
        for i_iter in tqdm(range(self.max_iter), desc=f'{running_name}'):
            self.opt_model.step()
            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())
            if i_iter % iters_per_print == 0 or i_iter == self.max_iter - 1:
                print(f'min energy: {self.opt_model.energy.min(dim=0)[0]:.4f}')
                print(f'min energy index: {self.opt_model.energy.min(dim=0)[1]}')
            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {f'{i_energy}': energy[i_energy] for i_energy in range(len(energy))}
                self.writer.add_scalars(main_tag=f'energy/{running_name}', tag_scalar_dict=tag_scaler_dict, global_step=i_iter)
                self.writer.add_scalar(tag=f'index/{running_name}', scalar_value=energy.index(min(energy)), global_step=i_iter)
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        return q_trajectory, self.opt_model.energy.detach().cpu().clone(), self.steps_per_iter
