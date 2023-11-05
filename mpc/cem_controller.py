import os 
import time
import pickle
import numpy as np
from PIL import Image, ImageDraw

import utils.plant_utils as plant_utils
import utils.utils as unet_utils
import matplotlib.pyplot as plt 
from mpc.gaussian_sampler import GaussianCEMSampler, TruncatedGaussianCEMSampler
from mpc.baseline_policy import Policy
from tqdm import tqdm

class Hparam:
    def __init__(self) -> None:
        pass

class CEMBaseController(Policy):
    def __init__(self, policyparams, envparams):
        Policy.__init__(self, policyparams, envparams)
        self._hp.update(self._default_hparams())
        sampler_class = policyparams.get('sampler', TruncatedGaussianCEMSampler)
        for name, value in self._additional_params().items():
            self._hp[name] = value
        for name, value in sampler_class.get_default_hparams().items():
            self._hp[name] = value
        for name, value in policyparams.items():
            self._hp[name] = value

        self._t = None
        self._n_iter = self._hp['iterations']

        away_from_top = self._hp['away_from_top']
        self.actionspace_limits[1][1] = self.actionspace_limits[1][1] - away_from_top
        x_length = self.actionspace_limits[0][1] - self.actionspace_limits[0][0]
        self._hp['initial_sampling_configurations']['x']['mean'] = float(x_length / 2)
        self._hp['initial_sampling_configurations']['x']['min'] = 0
        self._hp['initial_sampling_configurations']['x']['max'] = float(x_length)
        # self._hp['parameter_bounds']['x']['min'] = 0
        # self._hp['parameter_bounds']['x']['max'] = float(x_length)
        y_length = self.actionspace_limits[1][1] - self.actionspace_limits[1][0]
        self._hp['initial_sampling_configurations']['y']['mean'] = float(y_length / 2)
        self._hp['initial_sampling_configurations']['y']['min'] = 0
        self._hp['initial_sampling_configurations']['y']['max'] = float(y_length)
        # self._hp['parameter_bounds']['y']['min'] = 0
        # self._hp['parameter_bounds']['y']['max'] = float(y_length) + away_from_top


        self._sampler = None
        self._t_since_replan = None
        self._best_indices = None
        self._best_actions = None
        self._best_actions_processed = None
        self.plan_stat = {}

        self._new_space_revealed_map_after_all_actions = None
        self._all_probs = None
        self._all_thresholded = None
        self._reveal_map_xy_start = None 
        self._height_maps = None
        self._all_intermediate_space_revealed = None

        if self._hp["save_each_iteration"]:
            iteration_information_save_folder = os.path.join(self._hp["data_root"], "iteration_information")
            if not os.path.exists(iteration_information_save_folder):
                os.mkdir(iteration_information_save_folder)
            self.iteration_information_save_folder = iteration_information_save_folder

    def _default_hparams(self):
        default_dict = {
            'z_values' : None,
            'exhaustive' : True,
            'iterations': 10,
            'minimum_selection': 10,
            'num_samples': 200,
            'sampler': TruncatedGaussianCEMSampler,
            'selection_frac': 0.3, # specifcy which fraction of best samples to use to compute mean and var for next CEM iteration
            'replan_interval': 0,
            'away_from_top' : 0.0,
            'explore_horizontal' : False,
            'processed_actions_var_order' : ['x','y','z', 'theta', 'length'],
        }
        return default_dict
    
    def _additional_params(self):
        return {}

    def reset(self):
        self._best_indices = None
        self._best_actions = None
        self._best_actions_processed = None
        self._t_since_replan = None
        self._sampler = self._hp['sampler'](self._hp)
        self.plan_stat = {} #planning statistics
        self.space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
        self._new_space_revealed_map_after_all_actions = None
        self._all_probs = None
        self._all_thresholded = None
        self._reveal_map_xy_start = None 
        self._height_maps = None
        self._all_intermediate_space_revealed = None

    def _post_process_action(self, actions):
        """
        :param actions (M, nactions, adim)
        """
        assert actions.shape[-1] == len(self._hp['initial_sampling_configurations'])
        processed_actions = np.zeros((actions.shape[0], actions.shape[1], len(self._hp['processed_actions_var_order'])))

        processed_actions[:,:,0] = self.actionspace_limits[0][0] + (actions[:,:,0] + 0.5) * self._hp['initial_sampling_configurations']['x']['discrete']
        processed_actions[:,:,1] = self.actionspace_limits[1][1] - (actions[:,:,1] + 0.5) * self._hp['initial_sampling_configurations']['y']['discrete']
        if 'z_level' in self._hp['var_order']:
            processed_actions[:,:,2] = np.asarray(self._hp['z_values'])[actions[...,3].astype(int)]
        else:
            processed_actions[:,:,2] = self.actionspace_limits[2][0] + self.envparams['workspace_z_buffer']
        
        max_lengths = plant_utils.get_actions_parameter_range(processed_actions[:,:,0], processed_actions[:,:,1], actions[:,:,2], self.workspace_limits, in_radians=True)
        max_lengths = np.minimum(max_lengths, self.envparams['action_length_unit'])
        processed_actions[:,:,3] = actions[:,:,2]
        processed_actions[:,:,4] = max_lengths
        if 'z_level' in self._hp['processed_actions_var_order']:
            processed_actions[:,:,5] = actions[...,3]
        # assert np.all(processed_actions[:,:,0] >= self.workspace_for_reveal[0][0])
        # assert np.all(processed_actions[:,:,0] <= self.workspace_for_reveal[0][1])
        # assert np.all(processed_actions[:,:,1] >= self.workspace_for_reveal[1][0])
        # assert np.all(processed_actions[:,:,1] <= self.workspace_for_reveal[1][1])
        return processed_actions
    
    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        raise NotImplementedError
    
    def get_K(self):
        K = self._hp['minimum_selection']
        if self._hp['selection_frac']:
            K = max(int(self._hp['selection_frac'] * self._hp['num_samples']), self._hp['minimum_selection'])
        return K

    def perform_exhaustive(self):
        K = self.get_K()
        options = []
        for var_name in self._hp['var_order']:
            var_dict = self._hp['initial_sampling_configurations'][var_name]
            possible_values = int((var_dict['max'] - var_dict['min']) / var_dict['discrete'])
            if var_name in ['x','y','z_level']:
                possible_values += 1
                
            if var_name in ['theta']:
                if self.envparams['env_type'] == 0:
                    possible_values = 7
                all_values = np.arange(possible_values) * var_dict['discrete']
            
            else:
                all_values = np.arange(possible_values) 
            options.append(all_values)
        all_combs = np.meshgrid(*options)
        all_combs_flatten = []
        for comb in all_combs:
            all_combs_flatten.append(comb.reshape(-1,))
        # vx,vy,vtheta = np.meshgrid(*options)
        # actions = np.stack([vx.reshape(-1,), vy.reshape(-1,), vtheta.reshape(-1,)]).T
        actions = np.stack(all_combs_flatten).T
        print(":: Number of actions for exhaustive search: ", actions.shape)
        if self._hp['invalid_range']:
            x_invalid = np.logical_and(actions[:,0] >= self._hp['invalid_range'][0], \
                                    actions[:,0] <= self._hp['invalid_range'][1])
            y_invalid = np.logical_and(actions[:,1] >= self._hp['invalid_range'][2], \
                                    actions[:,1] <= self._hp['invalid_range'][3])
            in_invalid_range = np.logical_and(x_invalid, y_invalid)
            
            valid_actions = np.logical_not(in_invalid_range)
            actions = actions[valid_actions]
        # actions = actions[:300]
        print(":: (Not invalid ones) Number of actions for exhaustive search: ", actions.shape)
        actions = actions[:, np.newaxis, :]
        
        processed_actions = self._post_process_action(actions)


        print(":: Number of actions for exhaustive search: ", processed_actions.shape)
        
        all_scores = []
        total_secs = 0
        for start_idx in range(0, len(actions), 300):
            
            end_idx = min(len(actions), start_idx+300)
            tic = time.perf_counter()
            scores_idx = self.evaluate_rollouts(actions[start_idx:end_idx,:], processed_actions[start_idx:end_idx,:], 0)
            all_scores.append(scores_idx)
            toc = time.perf_counter()
            total_secs += toc - tic
            print(f"Evaluated all actions in {toc - tic:0.2f} seconds")
        print(f":: Total number of seconds {total_secs:0.2f}")
        scores = np.concatenate(all_scores)
        # assert scores.shape == (actions.shape[0],), "score shape should be (num_samples,)"
        
        indices_sorted = scores.argsort()[::-1]
        self._best_indices = indices_sorted[:K]
        self._best_actions = actions[self._best_indices]
        self._best_actions_processed = processed_actions[self._best_indices]
        self.plan_stat['processed_actions_not_selected_0'] = processed_actions[indices_sorted[K:]]
        self.plan_stat['processed_actions_best_0'] = self._best_actions_processed
        self.plan_stat['scores_itr_0'] = scores
        self.plan_stat['mean_itr_0'] = None
        self.plan_stat['cov_itr_0'] = None
        self.plan_stat['itr'] = 0
        if self._hp["save_each_iteration"]:
            itr = 0
            # save the space reveal map at the beginning
            current_space_revealed_map_path = os.path.join(self.iteration_information_save_folder, f"{self.i_tr:05}_{self._t:05}_{itr}_space_revealed_map.pkl")
            saves = [
                self._new_space_revealed_map_after_all_actions,
                self._all_probs, 
                self._all_thresholded, 
                self._reveal_map_xy_start, 
                self._height_maps,
                self._all_intermediate_space_revealed,
            ]
            with open(current_space_revealed_map_path, "wb+") as f:
                pickle.dump(saves,f)
            

        self._t_since_replan = 0

    def explore_horizontal(self, actions):
        if np.random.rand(1)[0] >= 0.5:
            mask = np.isclose(actions[:,:,2], 0.0, rtol=0.0, atol=np.radians(5)) # num samples * nactions
            # flip = np.random.rand(np.sum(mask)) >= 0.5 # num samples * nactions
            # [np.where(flip)]
            actions[:,:,2][np.where(mask)] = np.pi
        return actions
    
    def perform_CEM(self, state):
        K = self.get_K()
        actions = self._sampler.sample_initial_actions(self._t, self._hp['num_samples'], state)
        for itr in tqdm(range(self._n_iter)):
            if itr > 0 and self._hp['explore_horizontal']:
                actions = self.explore_horizontal(actions)
            # np.unique(actions[...,2].reshape(-1,), return_counts=True)
            
            processed_actions = self._post_process_action(actions)
            
            start_time = time.time()
            scores = self.evaluate_rollouts(actions, processed_actions, itr)
            end_time = time.time()
            time_taken = end_time - start_time
            print("evaluate_rollouts Time taken: {:.3f} seconds".format(time_taken))

            
            assert scores.shape == (actions.shape[0],), "score shape should be (num_samples,)"

            indices_sorted = scores.argsort()[::-1]
            self._best_indices = indices_sorted[:K]
            self._best_actions = actions[self._best_indices]
            self._best_actions_processed = processed_actions[self._best_indices]
            
            if self._hp["save_each_iteration"]:
                # save the space reveal map at the beginning
                current_space_revealed_map_path = os.path.join(self.iteration_information_save_folder, f"{self.i_tr:05}_{self._t:05}_{itr}_space_revealed_map.pkl")
                saves = [
                    self._new_space_revealed_map_after_all_actions,
                    self._all_probs, 
                    self._all_thresholded, 
                    self._reveal_map_xy_start, 
                    self._height_maps,
                    self._all_intermediate_space_revealed,
                ]
                with open(current_space_revealed_map_path, "wb+") as f:
                    pickle.dump(saves,f)
                

            self.plan_stat[f'processed_actions_not_selected_{itr}'] = processed_actions[indices_sorted[K:]]
            self.plan_stat[f'processed_actions_best_{itr}'] = self._best_actions_processed
            self.plan_stat['scores_itr_{}'.format(itr)] = scores
            self.plan_stat[f'mean_itr_{itr}'] = self._sampler._mean
            self.plan_stat[f'cov_itr_{itr}'] = self._sampler._sigma
            self.plan_stat['itr'] = itr
            
            if itr < self._n_iter - 1:
                start_time = time.time()
                actions = self._sampler.sample_next_actions(self._hp['num_samples'], self._best_actions.copy(), scores[self._best_indices].copy())
                end_time = time.time()
                time_taken = end_time - start_time
                print("Time taken: {:.3f} seconds".format(time_taken))


        self._t_since_replan = 0

    def act(self, t=None, i_tr=None, state=None):
        self._state = state
        self.i_tr = i_tr
        self._t = t
        if self._hp['exhaustive']:
            self.perform_exhaustive()
        else:
            if self._hp['replan_interval']:
                if self._t_since_replan is None or self._t_since_replan + 1 >= self._hp['replan_interval']:
                    self.perform_CEM(state)
                else:
                    self._t_since_replan += 1
            else:
                self.perform_CEM(state)
        action = self._best_actions[0, self._t_since_replan]
        action_processed = self._best_actions_processed[0, self._t_since_replan]

        if not self._hp['exhaustive']:
            if self._best_actions is not None:
                action_plan_slice = self._best_actions[:, min(self._t_since_replan + 1, self._hp['nactions'] - 1):]
                self._sampler.log_best_action(action, action_plan_slice)
            else:
                self._sampler.log_best_action(action, None)

        '''
        actions: 
            raw sampling value, t=0 of trajectory with highest score after CEM
        ** actions_processed: 
            processed action value (t=0 of best trajectory), x,y,z,theta,length
        best_actions_processed: 
            processed action of all time steps and all top-K trajectories, (K, nactions, 5)
        plan_stat:
            for each iteration, (processed) actions selected, those that are not selected, mean, cov, scores
            also the largest iteration number  
        space_revealed_map_before_action: 
            before this round of planning, what space is already revealed
        ? new_space_revealed_map:
            after executing t=0 of the best trajectory, what space will be revealed
        ? new_space_revealed_map_after_all_actions
        '''
        idx_in_dataset = self._best_indices[0] * self._hp['nactions']
        
        action_processed_flatten = action_processed.reshape(-1,)
        print("::!!!! Choose x: ", (action_processed_flatten[0] - self.actionspace_limits[0][0]) / 0.02)
        print("::!!!! Choose y: ", (self.actionspace_limits[1][1] - action_processed_flatten[1]) / 0.02)
        print("::!!!! Choose z: ", action_processed_flatten[2], np.degrees(action_processed_flatten[3]), action_processed_flatten[4], action_processed_flatten[5])
        
        if self._hp['exhaustive']:
            return {
                'actions':action, 
                'actions_processed' : action_processed, 
                "best_actions_processed" : self._best_actions_processed,
                'plan_stat':self.plan_stat, 
                'space_revealed_map_before_action' : np.copy(self.space_revealed_map),
                
            }
        return {
            'actions':action, 
            'actions_processed' : action_processed, 
            "best_actions_processed" : self._best_actions_processed,
            'plan_stat':self.plan_stat, 
            'space_revealed_map_before_action' : np.copy(self.space_revealed_map),
            # 'new_space_revealed_map' : new_space_revealed_map,
            # 'new_space_revealed_map_after_all_actions' : self._new_space_revealed_map_after_all_actions[self._best_indices[0]],
            "current_space_revealed_map" : self._all_intermediate_space_revealed[self._best_indices[0]], 
            # "current_space_revealed_map" : self._new_space_revealed_map_after_all_actions[self._best_indices[0]],
        }


class SimpleController(CEMBaseController):
    def __init__(self, policyparams, envparams):
        CEMBaseController.__init__(self, policyparams, envparams)
        self.is_forward_model = False
    
    def _additional_params(self):
        hp = {
            # 'bottom_rectangle_ratio' : 0.6
        }
        return hp
    
    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        """
        :param actions: array (num_samples, nactions, ?)
        :param cem_itr: int iteration of CEM
        :return scores: array (num_samples, )
        """
        num_samples, nactions, _ = processed_actions.shape
        endp = np.zeros((num_samples, nactions, 2))
        thetas = processed_actions[:,:,3]
        lengths = processed_actions[:,:,4]
        endp[:,:,0] = processed_actions[:,:,0] + np.cos(thetas) * lengths
        endp[:,:,1] = processed_actions[:,:,1] + np.sin(thetas) * lengths

        greater_than_90 = np.where(thetas > np.pi * 0.5) 
        less_than_90 = np.where(np.logical_not(thetas > np.pi * 0.5)) 

        pa_0_flat = np.zeros_like(processed_actions[:,:,0])
        pa_0_flat[greater_than_90] = processed_actions[:,:,0][greater_than_90] + self.envparams['grabber_width'] * 0.5
        pa_0_flat[less_than_90] = processed_actions[:,:,0][less_than_90] - self.envparams['grabber_width'] * 0.5
        pa_0_flat = pa_0_flat.reshape(-1,)

        ep_0_flat = np.zeros_like(endp[:,:,0])
        ep_0_flat[greater_than_90] = endp[:,:,0][greater_than_90] + self.envparams['grabber_width'] * 0.5
        ep_0_flat[less_than_90] = endp[:,:,0][less_than_90] - self.envparams['grabber_width'] * 0.5
        ep_0_flat = ep_0_flat.reshape(-1,)
        
        # pa_0_flat = processed_actions[:,:,0].reshape(-1,) - self.envparams['grabber_width'] * 0.5
        # ep_0_flat = endp[:,:,0].reshape(-1,) - self.envparams['grabber_width'] * 0.5
        ep_1_flat = endp[:,:,1].reshape(-1,)
        coords = np.hstack([
            np.hstack([pa_0_flat, ep_0_flat]).reshape(-1,1),
            np.hstack([ep_1_flat, np.zeros(num_samples * nactions)]).reshape(-1,1),
        ])
        x_coords, y_coords = plant_utils.get_coordinate_in_image(coords, self.workspace_for_reveal[0][0], self.workspace_for_reveal[1][0], self.space_revealed_map_w, self.space_revealed_map_h, self.img_input_resolution, restrict_to_within_image=True) 
        pa_0_pix, ep_0_pix = x_coords.reshape((2,-1))
        ep_1_pix,_ = y_coords.reshape((2,-1))
        pa_0_pix = pa_0_pix.reshape((num_samples, nactions))
        ep_0_pix = ep_0_pix.reshape((num_samples, nactions))
        ep_1_pix = ep_1_pix.reshape((num_samples, nactions))

        all_intermediate_space_revealed = []
        all_new_space_revealed = []
        all_scores = []
        for sample_idx in range(num_samples):
            current_space_revealed_maps = []
            current_space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
            for step_idx in range(nactions):
                
                x_min = pa_0_pix[sample_idx][step_idx]
                x_max = ep_0_pix[sample_idx][step_idx]
                if x_min == x_max:
                    continue
                y_bottom = ep_1_pix[sample_idx][step_idx]
                points = [(x_min, self.y_start_calculate), (x_max, y_bottom), (x_min, y_bottom)]
                im = Image.new('RGB', (self.space_revealed_map_w, self.space_revealed_map_h))
                draw = ImageDraw.Draw(im)
                draw.polygon(points, fill = (255,255,255)) 
                space_revealed_by_action_i = np.asarray(im)[:,:,0].astype(bool).astype(int)
                x_start = min(x_min, x_max)
                x_end = max(x_min, x_max)
                space_revealed_by_action_i[y_bottom:self.space_revealed_map_h, x_start:x_end] = 1
                current_space_revealed_map = np.maximum(space_revealed_by_action_i, current_space_revealed_map)
                current_space_revealed_maps.append(np.copy(current_space_revealed_map))
                # import pdb;pdb.set_trace()
            
            all_new_space_revealed.append(current_space_revealed_map)
            all_intermediate_space_revealed.append(current_space_revealed_maps)
            new_discovery = current_space_revealed_map - self.space_revealed_map
            new_discovery = new_discovery[self.y_start_calculate:self.y_end_calculate+1, self.x_start_calculate:self.x_end_calculate+1]
            score = new_discovery[new_discovery > 0].sum()
            all_scores.append(score)
        
        self._new_space_revealed_map_after_all_actions = np.stack(all_new_space_revealed)
        self._all_intermediate_space_revealed = all_intermediate_space_revealed
        return np.asarray(all_scores)    
        # triangle_side_y = self.workspace_limits[1][1] - processed_actions[:,:,1] - processed_actions[:,:,4] * np.sin(processed_actions[:,:,3])
        # triangle_side_x = np.abs(processed_actions[:,:,4] * np.cos(processed_actions[:,:,3]))
        # triangle_area = 0.5 * triangle_side_x * triangle_side_y
        # side_y_2 = self.workspace_limits[1][1] - self.workspace_limits[1][0] - triangle_side_y
        # rectangle_area = triangle_side_x * side_y_2 * self._hp["bottom_rectangle_ratio"]
        # volume = (triangle_area + rectangle_area) * 1.0
        # volume = np.sum(volume, axis=1)
        # return volume

class RadialSpaghettiController(CEMBaseController):
    def __init__(self, policyparams, envparams):
        CEMBaseController.__init__(self, policyparams, envparams)
        self.is_forward_model = False
        cx,cy,cz = self._hp['center']
        radius = self._hp['plant_radius']
        bbox_aligned_space = np.asarray([
            [cx-radius, cy+radius],
            [cx+radius, cy-radius],
        ])
        bbox_xs, bbox_ys = self.get_coord_in_reveal_map_fn(bbox_aligned_space)
        self.bbox = [bbox_xs[0], bbox_ys[0], bbox_xs[1], bbox_ys[1]]
    
    def _additional_params(self):
        hp = {
            'center' : np.array([-0.004, 0.643, -0.757]),
            'plant_radius' : 0.35,
            'away_from_center' : 0.08,
        }
        return hp
    
    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        """
        :param actions: array (num_samples, nactions, ?)
        :param cem_itr: int iteration of CEM
        :return scores: array (num_samples, )
        """
        cx,cy,cz = self._hp['center']
        num_samples, nactions, _ = processed_actions.shape
        endp = np.zeros((num_samples, nactions, 2))
        thetas = processed_actions[:,:,3]
        lengths = processed_actions[:,:,4]
        endp[:,:,0] = processed_actions[:,:,0] + np.cos(thetas) * lengths
        endp[:,:,1] = processed_actions[:,:,1] + np.sin(thetas) * lengths

        second_third_quadrant = np.logical_and(thetas > np.pi * 0.5, thetas < np.pi * 1.5)
        greater_than_90 = np.where(second_third_quadrant)
        less_than_90 = np.where(np.logical_not(second_third_quadrant)) 

        pa_0_flat = np.zeros_like(processed_actions[:,:,0])
        pa_0_flat[greater_than_90] = processed_actions[:,:,0][greater_than_90] #+ self.envparams['grabber_width'] * 0.5
        pa_0_flat[less_than_90] = processed_actions[:,:,0][less_than_90] #- self.envparams['grabber_width'] * 0.5

        ep_0_flat = np.zeros_like(endp[:,:,0])
        ep_0_flat[greater_than_90] = endp[:,:,0][greater_than_90] #+ self.envparams['grabber_width'] * 0.5
        ep_0_flat[less_than_90] = endp[:,:,0][less_than_90] #- self.envparams['grabber_width'] * 0.5
        
        # pa_0_flat = processed_actions[:,:,0].reshape(-1,) - self.envparams['grabber_width'] * 0.5
        # ep_0_flat = endp[:,:,0].reshape(-1,) - self.envparams['grabber_width'] * 0.5
        start_1_flat = processed_actions[:,:,1]
        ep_1_flat = endp[:,:,1]
        
        all_intermediate_space_revealed = []
        all_new_space_revealed = []
        all_scores = []
        for sample_idx in range(num_samples):
            current_space_revealed_maps = []
            current_space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
            for step_idx in range(nactions):
                im = Image.new('RGB', (self.space_revealed_map_w, self.space_revealed_map_h))
                draw = ImageDraw.Draw(im)
                
                x_start = pa_0_flat[sample_idx][step_idx]
                x_end = ep_0_flat[sample_idx][step_idx]
                y_start = start_1_flat[sample_idx][step_idx]
                y_end = ep_1_flat[sample_idx][step_idx]
                start_dist_from_center = np.linalg.norm(np.array([x_start, y_start, cz]) - self._hp['center'])
                end_dist_from_center = np.linalg.norm(np.array([x_end, y_end, cz]) - self._hp['center'])
                if start_dist_from_center >= self._hp['away_from_center'] and end_dist_from_center >= self._hp['away_from_center']:
                    theta_s = np.arctan2(y_start - cy , x_start - cx + 1e-5)
                    if theta_s > 0:
                        theta_s -= np.pi * 2
                    theta_e = np.arctan2(y_end - cy, x_end - cx + 1e-5)
                    if theta_e > 0:
                        theta_e -= np.pi * 2
                    theta_s = -theta_s
                    theta_e = -theta_e
                    
                    if np.abs(theta_s - theta_e) > np.abs(theta_s - theta_e + 2 * np.pi):
                        diff = np.abs(theta_s - theta_e + 2 * np.pi)
                        if theta_s - theta_e + 2 * np.pi > 0:
                            start_degree = np.degrees(theta_e)
                            end_degree = np.degrees(theta_s + 2 * np.pi)
                        else:
                            start_degree = np.degrees(theta_s + 2 * np.pi)
                            end_degree = np.degrees(theta_e)
                    else:
                        diff = np.abs(theta_s - theta_e)
                        if theta_s - theta_e > 0:
                            start_degree = np.degrees(theta_e)
                            end_degree = np.degrees(theta_s)
                        else:
                            start_degree = np.degrees(theta_s)
                            end_degree = np.degrees(theta_e)


                    if diff <= np.pi * 0.5:
                        # print(np.degrees(diff))
                        draw.pieslice(self.bbox, start=start_degree, end=end_degree, fill = (255,255,255))

                space_revealed_by_action_i = np.asarray(im)[:,:,0].astype(bool).astype(int)
                
                current_space_revealed_map = np.maximum(space_revealed_by_action_i, current_space_revealed_map)
                current_space_revealed_maps.append(np.copy(current_space_revealed_map))
            
            all_new_space_revealed.append(current_space_revealed_map)
            all_intermediate_space_revealed.append(current_space_revealed_maps)
            new_discovery = current_space_revealed_map - self.space_revealed_map
            new_discovery = new_discovery[self.y_start_calculate:self.y_end_calculate+1, self.x_start_calculate:self.x_end_calculate+1]
            score = new_discovery[new_discovery > 0].sum()
            all_scores.append(score)
        
        self._new_space_revealed_map_after_all_actions = np.stack(all_new_space_revealed)
        self._all_intermediate_space_revealed = all_intermediate_space_revealed
        return np.asarray(all_scores) 

class RadialSpaghettiCameraFrameController(CEMBaseController):
    def __init__(self, policyparams, envparams):
        CEMBaseController.__init__(self, policyparams, envparams)
        self.is_forward_model = False
        radius = self._hp['plant_radius']
        center_z = np.copy(self._hp['center'])
        center_z[2] = self._hp['z_values'][-1]
        cx,cy,cz = center_z
        bbox_aligned_space = np.asarray([
            [cx-radius, cy+radius, cz],
            [cx+radius, cy-radius, cz],
            [cx,cy,cz],
        ])
        bbox_xs, bbox_ys = self.project_aligned_frame_pts_to_img(bbox_aligned_space)
        self.bbox = [bbox_xs[0], bbox_ys[0], bbox_xs[1], bbox_ys[1]]
        self.center_x, self.center_y = bbox_xs[2],bbox_ys[2]
        self.center_img_coord = np.asarray([bbox_xs[2],bbox_ys[2]])
        self._hp['bbox'] = np.asarray(self.bbox).astype(int).tolist()
        self._hp['center_x'] = self.center_x
        self._hp['center_y'] = self.center_y
    
    def _additional_params(self):
        hp = {
            'center' : np.array([-0.004, 0.643, -0.757]),
            'plant_radius' : 0.35,
            'z_values' : [-1.2221, -1.1721033028632442, -1.1221033028632441],
        }
        return hp
    
    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        """
        :param actions: array (num_samples, nactions, ?)
        :param cem_itr: int iteration of CEM
        :return scores: array (num_samples, )
        """
        
        num_samples, nactions, _ = processed_actions.shape
        startp = np.copy(processed_actions[...,:3])
        
        endp = np.zeros((num_samples, nactions, 3))
        thetas = processed_actions[:,:,3]
        lengths = processed_actions[:,:,4]
        endp[:,:,0] = processed_actions[:,:,0] + np.cos(thetas) * lengths
        endp[:,:,1] = processed_actions[:,:,1] + np.sin(thetas) * lengths
        endp[:,:,2] = processed_actions[:,:,2]

        second_third_quadrant = np.logical_and(thetas > np.pi * 0.5, thetas < np.pi * 1.5)
        greater_than_90 = np.where(second_third_quadrant)
        less_than_90 = np.where(np.logical_not(second_third_quadrant)) 

        startp[...,0][greater_than_90] = processed_actions[...,0][greater_than_90] + self.envparams['grabber_width'] * 0.5
        startp[...,0][less_than_90] = processed_actions[...,0][less_than_90] - self.envparams['grabber_width'] * 0.5
        endp[...,0][greater_than_90] = endp[...,0][greater_than_90] + self.envparams['grabber_width'] * 0.5
        endp[...,0][less_than_90] = endp[...,0][less_than_90] - self.envparams['grabber_width'] * 0.5
        
        startp_flat = startp.reshape((-1,3))
        start_xs, start_ys = self.project_aligned_frame_pts_to_img(startp_flat)
        start_xs = start_xs.reshape((num_samples, nactions))
        start_ys = start_ys.reshape((num_samples, nactions))

        endp_flat = endp.reshape((-1,3))
        end_xs,end_ys = self.project_aligned_frame_pts_to_img(endp_flat)
        end_xs = end_xs.reshape((num_samples, nactions))
        end_ys = end_ys.reshape((num_samples, nactions))
        
        all_intermediate_space_revealed = []
        all_new_space_revealed = []
        all_scores = []
        for sample_idx in range(num_samples):
            current_space_revealed_maps = []
            current_space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
            for step_idx in range(nactions):
                im = Image.new('RGB', (self.space_revealed_map_w, self.space_revealed_map_h))
                draw = ImageDraw.Draw(im)
                
                x_start = start_xs[sample_idx][step_idx]
                x_end = end_xs[sample_idx][step_idx]
                y_start = start_ys[sample_idx][step_idx]
                y_end = end_ys[sample_idx][step_idx]
                
                theta_s = np.arctan2(y_start - self.center_y , x_start - self.center_x + 1e-5)
                if theta_s > 0:
                    theta_s -= np.pi * 2
                theta_e = np.arctan2(y_end - self.center_y, x_end - self.center_x + 1e-5)
                if theta_e > 0:
                    theta_e -= np.pi * 2
                # theta_s = -theta_s
                # theta_e = -theta_e
                
                if np.abs(theta_s - theta_e) > np.abs(theta_s - theta_e + 2 * np.pi):
                    diff = np.abs(theta_s - theta_e + 2 * np.pi)
                    if theta_s - theta_e + 2 * np.pi > 0:
                        start_degree = np.degrees(theta_e)
                        end_degree = np.degrees(theta_s + 2 * np.pi)
                    else:
                        start_degree = np.degrees(theta_s + 2 * np.pi)
                        end_degree = np.degrees(theta_e)
                else:
                    diff = np.abs(theta_s - theta_e)
                    if theta_s - theta_e > 0:
                        start_degree = np.degrees(theta_e)
                        end_degree = np.degrees(theta_s)
                    else:
                        start_degree = np.degrees(theta_s)
                        end_degree = np.degrees(theta_e)


                if diff <= np.pi * 0.5:
                    # print(np.degrees(diff))
                    draw.pieslice(self.bbox, start=start_degree, end=end_degree, fill = (255,255,255))

                space_revealed_by_action_i = np.asarray(im)[:,:,0].astype(bool).astype(int)
                
                current_space_revealed_map = np.maximum(space_revealed_by_action_i, current_space_revealed_map)
                current_space_revealed_maps.append(np.copy(current_space_revealed_map))
            
            all_new_space_revealed.append(current_space_revealed_map)
            all_intermediate_space_revealed.append(current_space_revealed_maps)
            new_discovery = current_space_revealed_map - self.space_revealed_map
            new_discovery = new_discovery[self.y_start_calculate:self.y_end_calculate+1, self.x_start_calculate:self.x_end_calculate+1]
            score = new_discovery[new_discovery > 0].sum()
            all_scores.append(score)
        
        self._new_space_revealed_map_after_all_actions = np.stack(all_new_space_revealed)
        self._all_intermediate_space_revealed = all_intermediate_space_revealed
        return np.asarray(all_scores)       