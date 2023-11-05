import cv2
import numpy as np
import utils.plant_utils as plant_utils

from autolab_core import YamlConfig, RigidTransform, Point, PointCloud
from perception import CameraIntrinsics

class Policy(object):
    def __init__(self, policyparams, envparams):
        
        self.envparams = envparams
        self.actionspace_limits = np.asarray(self.envparams['workspace_limits']) # sampling range
        self.workspace_limits = np.asarray(self.envparams['workspace_limits']) # how far the robot can push
        self._hp = {
            'is_forward_model' : False,
            'img_input_resolution' : 0.002,
            'height_diff_threshold' : -0.05,
            'minimum_decrease_before_transformation' : -0.05,
            'close_to_board' : False,
            'dataset_from_camera_frame' : False,
        }
        for name, value in policyparams.items():
            self._hp[name] = value

        self.T_flip = RigidTransform(rotation=np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]),translation=np.array([0.0,0.0,0.0]), from_frame='azure_kinect_overhead', to_frame='azure_kinect_overhead_flipped')  
        self.T_camera_cameraupright = RigidTransform.load(envparams['plant_upright_transformation_path'])
        self.T_camera_cameraupright_height = RigidTransform.load(envparams['plant_upright_transformation_path_height'])
        self.T_camera_world = RigidTransform.load(envparams['extrinsics_file_path'])   
        self.T_sampled_action_space_to_world = self.T_camera_world * self.T_flip.inverse() * self.T_camera_cameraupright.inverse()
        self.intrinsic_iam = CameraIntrinsics.load(envparams['intrinsics_file_path'])
        self._hp['plant_upright_transformation_path'] = envparams['plant_upright_transformation_path']
        self._hp['plant_upright_transformation_path_height'] = envparams['plant_upright_transformation_path_height']
        self._hp['extrinsics_file_path'] = envparams['extrinsics_file_path']
        if envparams['use_rs']:
            self._hp['intrinsics_file_rs_path'] = envparams['intrinsics_file_rs_path']
            self._hp['rs_to_ee_path'] = envparams['rs_to_ee_path']

        if self._hp['dataset_from_camera_frame']:
            space_revealed_map_w = envparams['space_revealed_map_w']
            space_revealed_map_h = envparams['space_revealed_map_h']
            x_coords = envparams['calculate_x_bound_in_reveal_map']
            y_coords = envparams['calculate_y_bound_in_reveal_map']
            x_coords_action = envparams['action_x_bound_in_reveal_map']
            y_coords_action = envparams['action_y_bound_in_reveal_map']
        else:
            self.img_input_resolution = self._hp['img_input_resolution']
            self.workspace_for_reveal_original = np.asarray(self.envparams["workspace_for_reveal"])
            self.workspace_for_reveal = np.asarray(self.envparams['global_workspace_limits'])
            space_revealed_map_h, space_revealed_map_w = plant_utils.from_workspace_limits_to_2d_dimension(self.workspace_for_reveal,self.img_input_resolution)
        
            self.four_corners_global_in_image = np.asarray(self.envparams['four_corners_global_in_image'])
            self._hp["workspace_for_reveal"] = self.workspace_for_reveal.astype(float).tolist()

            dst_pts = np.asarray([
                [0, space_revealed_map_h-1],
                [0, 0],
                [space_revealed_map_w-1, space_revealed_map_h-1],
                [space_revealed_map_w-1, 0],
            ])
            self.homography_matrix,_ = cv2.findHomography(self.four_corners_global_in_image.astype(np.float32), dst_pts.astype(np.float32))
            print(self.homography_matrix)

            self.get_coord_in_reveal_map_fn = lambda x : plant_utils.get_coordinate_in_image(x, self.workspace_for_reveal[0][0], self.workspace_for_reveal[1][0], self.space_revealed_map_w, self.space_revealed_map_h, self.img_input_resolution, restrict_to_within_image=True)

            x_coords, y_coords = plant_utils.get_coordinate_in_image(self.workspace_for_reveal_original[:2].T, self.workspace_for_reveal[0][0], self.workspace_for_reveal[1][0], self.space_revealed_map_w, self.space_revealed_map_h, self.img_input_resolution, restrict_to_within_image=True)

            x_coords_action, y_coords_action = plant_utils.get_coordinate_in_image(self.workspace_limits[:2].T, self.workspace_for_reveal[0][0], self.workspace_for_reveal[1][0], self.space_revealed_map_w, self.space_revealed_map_h, self.img_input_resolution, restrict_to_within_image=True)
        
        self._hp["space_revealed_map_w"] = int(space_revealed_map_w)
        self._hp["space_revealed_map_h"] = int(space_revealed_map_h)
        self.space_revealed_map_w = space_revealed_map_w
        self.space_revealed_map_h = space_revealed_map_h
        self.space_revealed_map = np.zeros((space_revealed_map_h, space_revealed_map_w))

        self.x_start_calculate, self.x_end_calculate = x_coords
        self.y_end_calculate, self.y_start_calculate = y_coords

        self.x_start_action, self.x_end_action = x_coords_action
        self.y_end_action, self.y_start_action = y_coords_action
        
        self._hp["calculate_bounds_in_space_revealed_map"] = np.asarray([self.x_start_calculate, self.x_end_calculate, self.y_start_calculate, self.y_end_calculate]).astype(int).tolist()
        self._hp["action_bounds_in_space_revealed_map"] = np.asarray([self.x_start_action, self.x_end_action, self.y_start_action, self.y_end_action]).astype(int).tolist()

    def project_aligned_frame_pts_to_img(self, pts):
        assert pts.shape[1] == 3
        pts = PointCloud(np.asarray(pts).T, frame = "azure_kinect_overhead_upright")
        pts_camera = self.T_flip.inverse() * self.T_camera_cameraupright.inverse() * pts
        x_coords, y_coords = self.intrinsic_iam.project(pts_camera).data
        return x_coords, y_coords  
    
    def project_pts_to_img(self, pts):
        assert pts.shape[1] == 3
        pts = PointCloud(np.asarray(pts).T, frame="world")
        pts_camera = self.T_camera_world.inverse() * pts
        x_coords, y_coords = self.intrinsic_iam.project(pts_camera).data
        return x_coords, y_coords

    def act(self, *args):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        self.space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))

class TilingPolicy(Policy):
    def __init__(self, policyparams, envparams):
        Policy.__init__(self, policyparams, envparams)
        self._hp.update(self._default_hparams())
        for name, value in policyparams.items():
            self._hp[name] = value
        self.is_forward_model = self._hp['is_forward_model']
        possible_xs = np.arange(self.workspace_limits[0][0], self.workspace_limits[0][1], self._hp['bin_width']) + 0.03
        bin_height = (self.workspace_limits[1][1] - self.workspace_limits[1][0]) / 5
        print(bin_height)
        possible_ys = np.arange(self.workspace_limits[1][0], self.workspace_limits[1][1], bin_height)[1:]
        # if np.abs(possible_ys[-1] - self.workspace_limits[1][1]) >= 0.1:
        # possible_ys = np.concatenate([possible_ys, np.array([self.workspace_limits[1][1]])])
        possible_xs = np.concatenate([possible_xs, np.array([self.workspace_limits[0][1]-0.03])])
        vx,vy = np.meshgrid(possible_xs, possible_ys)
        ptx_pty = np.stack([vx.reshape(-1,), vy.reshape(-1,)]).T
        all_actions = np.zeros((len(ptx_pty), 5))
        all_actions[:,:2] = ptx_pty
        all_actions[:,2] = self.workspace_limits[2][0] + self.envparams['workspace_z_buffer']
        angles = np.repeat(np.array([0.0] * (len(possible_xs)-1) + [np.pi]).reshape(1,-1),len(possible_ys),axis=0).reshape(-1,)
        all_actions[:,3] = angles
        right_len = np.minimum(self.workspace_limits[0][1] - possible_xs[:-1], self.envparams['action_length_unit'])
        left_len = self.envparams['action_length_unit']
        lengths = np.repeat(np.concatenate([right_len, np.array([left_len])]).reshape(1,-1),len(possible_ys),axis=0).reshape(-1,)
        all_actions[:,4] = lengths
        self.num_tiles = np.arange(len(all_actions))
        print("Number of tiles: ", len(all_actions))
        print(all_actions)
        np.random.shuffle(self.num_tiles)
        self.count = 0
        self.all_actions = all_actions
        self._hp['num_tiles'] = self.num_tiles
        self._hp['all_actions'] = all_actions.astype(float).tolist()

    def reset(self):
        self.space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
        np.random.shuffle(self.num_tiles)
        self.count = 0
    
    def _default_hparams(self):
        default_dict = {
            'bin_width' : 0.15,
            # 'bin_height' : 0.15,
        }
        return default_dict

    def act(self, t=None, i_tr=None, state=None):
        print("Tile: ", self.num_tiles[self.count])
        actions_processed = self.all_actions[self.num_tiles[self.count]]
        self.count += 1
        if self.count >= len(self.num_tiles):
            np.random.shuffle(self.num_tiles)
            self.count = 0
        return {
            'actions_processed' : actions_processed,
            'space_revealed_map_before_action' : np.copy(self.space_revealed_map),
        }

import random
def incremental_farthest_search(points, k):
    def distance(pt1,pt2):
        return np.linalg.norm(pt1[:3]-pt2[:3])
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(random.randint(0, len(remaining_points) - 1)))
    for _ in range(k-1):
        distances = [distance(p, solution_set[0]) for p in remaining_points]
        for i, p in enumerate(remaining_points):
            for j, s in enumerate(solution_set):
                distances[i] = min(distances[i], distance(p, s))
        solution_set.append(remaining_points.pop(distances.index(max(distances))))
    return solution_set

class RandomPolicy(Policy):
    def __init__(self, policyparams, envparams):
        Policy.__init__(self, policyparams, envparams)
        self._hp.update(self._default_hparams())
        for name, value in policyparams.items():
            self._hp[name] = value
        
        x_length = self.actionspace_limits[0][1] - self.actionspace_limits[0][0]
        self.x_possible_values = np.ceil(x_length / 0.02).astype(int)
        y_length = self.actionspace_limits[1][1] - self.actionspace_limits[1][0]
        self.y_possible_values = np.ceil(y_length / 0.02).astype(int)
        self.angle_list_all = np.arange(8) * (np.pi/4)
        print(":: Random policy: ", self.x_possible_values, self.y_possible_values, self.angle_list_all)
    
    def _default_hparams(self):
        default_dict = {
            'invalid_range' : None,
            'z_values' : [],
        }
        return default_dict
    
    def act(self, t=None, i_tr=None, state=None):

        actions_processed = None
        while actions_processed is None:
            pix_x = np.random.choice(self.x_possible_values)
            pix_y = np.random.choice(self.y_possible_values)
            z_level = np.random.choice(len(self._hp['z_values']))
            while self._hp['invalid_range'] is not None and pix_x >= self._hp['invalid_range'][0] and pix_x <= self._hp['invalid_range'][1] and pix_y >= self._hp['invalid_range'][2] and pix_y <= self._hp['invalid_range'][3]:
                pix_x = np.random.choice(self.x_possible_values)
                pix_y = np.random.choice(self.y_possible_values)

            np.random.shuffle(self.angle_list_all)
            angle_idx = 0
            pt_x = self.actionspace_limits[0][0] + (pix_x + 0.5) * 0.02
            pt_y = self.actionspace_limits[1][1] - (pix_y + 0.5) * 0.02
            # pt_z = workspace_limits[2][0] + 0.2 + z_level * 0.05
            pt_z = self._hp['z_values'][z_level]        
            
            length = -1
            while True:
                if angle_idx >= len(self.angle_list_all):
                    break
                angle_list, max_length_list, end_xy_points = plant_utils.get_action_parameter_range(pt_x, pt_y, self.actionspace_limits, angle_list=[self.angle_list_all[angle_idx]], in_radians=True)
                length = min(max_length_list[0], self.envparams['action_length_unit'])
                if length >= 0.1:
                    actions_processed = np.asarray([pt_x, pt_y, pt_z, self.angle_list_all[angle_idx], length, z_level])
                    break
                else:
                    angle_idx += 1
        print(":: actions_processed in random policy: ", actions_processed)
        return {
            'actions_processed' : actions_processed,
            'space_revealed_map_before_action' : np.copy(self.space_revealed_map),
        }

class RadialTilingPolicy(Policy):
    def __init__(self, policyparams, envparams):
        Policy.__init__(self, policyparams, envparams)
        self._hp.update(self._default_hparams())
        for name, value in policyparams.items():
            self._hp[name] = value
        self.cx,self.cy,_ = self._hp['center']
        if self._hp['z_values'] is None:
            self.cz_list = [self.actionspace_limits[2][0] + self.envparams['workspace_z_buffer']]
        else:
            self.cz_list = sorted(self._hp['z_values'])
        
        self.set_up_concentric_circles()    
    
    def set_up_concentric_circles(self):
        action_width, action_height = self.workspace_limits[:2,1] - self.workspace_limits[:2,0]
        if type(self._hp['radius']) is int:
            num_circles = int(max(action_width, action_height) / (2 * self._hp['radius']))
            all_ri = [self._hp['radius'] * (i+1) for i in range(num_circles)]
        else:
            num_circles = len(self._hp['radius'])
            all_ri = self._hp['radius']
        start_points = []
        prop_acc = 1
        for i in range(num_circles):
            ri = all_ri[i]
            if i > 0:
                prop = all_ri[i-1] / ri
                prop_acc *= prop
            thetai = self._hp['theta'] * prop_acc #/ (2 ** i)
            num_points = np.ceil(2 * np.pi / thetai).astype(int)
            for j in range(num_points):
                x_start = self.cx + ri * np.cos(j * thetai)
                y_start = self.cy + ri * np.sin(j * thetai)
                for z_level, cz in enumerate(self.cz_list):
                    start_points.append([x_start, y_start, cz, j * thetai, ri, int(z_level)])
                    break
        start_points = np.asarray(start_points)
        x_in_range = np.logical_and(start_points[:,0] >= self.actionspace_limits[0][0], \
                                    start_points[:,0] <= self.actionspace_limits[0][1])
        y_in_range = np.logical_and(start_points[:,1] >= self.actionspace_limits[1][0], \
                                    start_points[:,1] <= self.actionspace_limits[1][1])
        in_range = np.logical_and(x_in_range, y_in_range)
        if self._hp['invalid_range']:
            x_indices = np.floor((start_points[:,0] - self.actionspace_limits[0][0]) / 0.02)
            y_indices = np.floor((self.actionspace_limits[1][1] - start_points[:,1]) / 0.02)
            
            x_invalid = np.logical_and(x_indices >= self._hp['invalid_range'][0], \
                                    x_indices <= self._hp['invalid_range'][1])
            y_invalid = np.logical_and(y_indices >= self._hp['invalid_range'][2], \
                                    y_indices <= self._hp['invalid_range'][3])
            in_invalid_range = np.logical_and(x_invalid, y_invalid)
            
            in_range = np.logical_and(in_range, np.logical_not(in_invalid_range))
        start_points = start_points[in_range]
        action_xs = np.concatenate([start_points[:,0], start_points[:,0]])
        action_ys = np.concatenate([start_points[:,1], start_points[:,1]])
        action_zs = np.concatenate([start_points[:,2], start_points[:,2]])
        angles = np.concatenate([start_points[:,3] + np.pi/2, start_points[:,3] - np.pi/2])
        action_z_levels = np.concatenate([start_points[:,5], start_points[:,5]])
        max_lengths = plant_utils.get_actions_parameter_range(action_xs, 
                                                            action_ys, 
                                                            angles, self.workspace_limits, in_radians=True)
        max_lengths = np.minimum(max_lengths, self.envparams['action_length_unit'])
        max_lengths_not_too_small = max_lengths >= 0.1
        # start_points = np.hstack([start_points, max_lengths.reshape(-1,1)])
        # #theta2s = np.arctan(max_lengths / start_points[:,-1])
        # for idx in enumerate(start_points):
        #     action = start_points[idx]
        #     # theta2 = theta2s[idx]
        #     x_end = action[0] + np.cos(angles[idx]) * max_lengths[idx]
        #     y_end = action[1] + np.sin(angles[idx]) * max_lengths[idx]
        all_original_actions = np.hstack([
            action_xs[max_lengths_not_too_small].reshape(-1,1),
            action_ys[max_lengths_not_too_small].reshape(-1,1),
            action_zs[max_lengths_not_too_small].reshape(-1,1),
            # np.ones(len(action_xs[max_lengths_not_too_small])).reshape(-1,1) * self.cz,
            angles[max_lengths_not_too_small].reshape(-1,1),
            max_lengths[max_lengths_not_too_small].reshape(-1,1),
            action_z_levels[max_lengths_not_too_small].reshape(-1,1),
        ])
        
        if self._hp['spread_out']:
            if self._hp['continue_t'] + 1 > 0:
                all_actions = np.asarray(self._hp['all_actions'])
            else:
                np.random.shuffle(all_original_actions)
                all_actions = incremental_farthest_search(list(np.copy(all_original_actions)), self._hp['T'])
                all_actions = np.asarray(all_actions)
        else:
            all_actions = all_original_actions
        self.all_original_actions = all_original_actions
        self.num_tiles = np.arange(len(all_actions))
        print("Number of tiles: ", len(all_actions))
        if not self._hp['spread_out']:
            np.random.shuffle(self.num_tiles)
        self.count = self._hp['continue_t'] + 1 
        self.all_actions = all_actions
        self._hp['num_tiles'] = self.num_tiles
        self._hp['all_actions'] = all_actions.astype(float).tolist()
        

    def _default_hparams(self):
        default_dict = {
            'invalid_range' : None,
            'radius' : [0.15, 0.2, 0.25, 0.3],
            'theta' : np.pi / 4,
            'arc_length' : 0.15,
            'center' : np.array([-0.02902323, 0.60712, -0.7484]),
            'spread_out' : False,
            'z_values' : None,
        }
        return default_dict

    def reset(self):
        self.space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
        if self._hp['spread_out']:
            np.random.shuffle(self.all_original_actions)
            all_actions = incremental_farthest_search(list(np.copy(self.all_original_actions)), self._hp['T'])
            self.all_actions = np.asarray(all_actions)
        else:
            np.random.shuffle(self.num_tiles)
        self.count = 0
    
    def act(self, t=None, i_tr=None, state=None):
        print("Tile: ", self.num_tiles[self.count])
        actions_processed = self.all_actions[self.num_tiles[self.count]]
        self.count += 1
        if self.count >= len(self.num_tiles):
            if self._hp['spread_out']:
                all_actions = incremental_farthest_search(list(np.copy(self.all_original_actions)), self._hp['T'])
                all_actions = np.asarray(all_actions)
                self.all_actions = all_actions
            else:
                np.random.shuffle(self.num_tiles)
            self.count = 0
        print("actions_processed: ", actions_processed)
        return {
            'actions_processed' : actions_processed,
            'space_revealed_map_before_action' : np.copy(self.space_revealed_map),
        }