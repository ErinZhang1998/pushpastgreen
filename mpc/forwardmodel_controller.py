import os 
import pickle
import cv2

import time 

import open3d as o3d
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .cem_controller import CEMBaseController

import sys
sys.path.append("/home/zhang401local/Documents/plant_pointcloud/unet")
import utils.plant_utils as plant_utils
import utils.metrics as metrics_utils
from utils.data_loading import calculate_action_end, normalize_height_map
from unet.unet_model import UNet
import copy
from tqdm import tqdm

from autolab_core import YamlConfig, RigidTransform, Point, PointCloud
from perception import CameraIntrinsics

class ForwardModelController(CEMBaseController):
    def __init__(self, policyparams, envparams):
        CEMBaseController.__init__(self, policyparams, envparams)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theta_sampling_config = self._hp['initial_sampling_configurations']['theta']
        self.img_input_w = self.img_input_h = np.floor((self._hp['img_input_distance_to_boundary'] * 2) / self.img_input_resolution).astype(int)
        
        if self._hp['close_to_board']:
            self._hp['threshold_class'] = 1

        self.num_angles = self._hp['num_angles']
        # 1 + int((self.theta_sampling_config['max'] - self.theta_sampling_config['min']) / self.theta_sampling_config['discrete'])

        # initialize forward model here.
        if self._hp['use_predict_precondition']:
            hp2 = copy.deepcopy(self._hp)
            hp2['predict_precondition'] = True
            self.precond_predictor = UNet(hp2)
            self.precond_predictor.load_state_dict(torch.load(self._hp['precondition_load'], map_location=self.device))
            self.precond_predictor.to(device=self.device)
            
        self._hp['predict_precondition'] = False
        self.predictor = UNet(self._hp)
        self.predictor.load_state_dict(torch.load(self._hp['load'], map_location=self.device))
        self.predictor.to(device=self.device)

        self.image_tensor_acc = []
        self.height_tensor_acc = []
        self.img_mask_acc = []
        self.action_info_acc = []
        self.gt_mask_acc = [] # getting after env.step
        self.height_diff_acc = [] # getting after env.step
        self.postaction_mask_acc = [] # getting after env.step
        self.is_forward_model = self._hp['is_forward_model']
        if self._hp['adaptive_train']:
            self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-5, weight_decay=1e-8)
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            if len(self._hp['classes']) > 1:
                self.criterion1 = nn.NLLLoss(reduction='none')
            if self._hp['use_regression_loss']:
                if self._hp['huber_loss_delta']:
                    self.criterion2 = nn.HuberLoss(reduction='none', delta=self._hp['huber_loss_delta'])
                else:
                    self.criterion2 = nn.MSELoss(reduction='none')
    
    def reset(self):
        super(ForwardModelController, self).reset()
        self.image_tensor_acc = []
        self.height_tensor_acc = []
        self.img_mask_acc = []
        self.action_info_acc = []
        self.gt_mask_acc = [] # getting after env.step
        self.height_diff_acc = [] # getting after env.step
        self.postaction_mask_acc = [] # getting after env.step

    def _additional_params(self):
        hp = {
            'is_forward_model' : True,
            'adaptive_train' : False,
            'num_angles' : 7,
            'img_input_distance_to_boundary' : 0.15,
            'distance_to_boundary' : 0.15,
            'iou_threshold' : 0.5,
            'use_height_feature' : True,
            'no_rgb_feature' : False,
            'action_as_image' : False,
            'no_image' : False,
            'no_action' : False,
            'use_coordinate_feature' : False, # TODO
            'classes' : [-0.03],
            'threshold_class' : 1,
            'huber_loss_delta' : 1.0,
            'use_regression_loss' : False,
            'use_postaction_mask_loss' : False,
            'upconv_channels' : [512, 512, 256, 256, 128, 128, 64, 64],
            'use_resnet' : False,
            'resnet_maxpool_after_enc' : False,
            'downconv_channels' : [64, 64, 128, 256, 512],
            'unet_downsample_layer_type' : 'DownMaxpoolSingleConv',
            'action_upconv_channels' : [64, 64, 64, 128, 128, 128, 128],
            'action_upconv_sizes' : [2,4,9],
            'action_input_dim' : 9,
            'load' : "/scratch0/zhang401local/plant_model/prosperous-dragon-73/checkpoint_step_1400.pth",
            'batch_size' : 10,
            'rotation_augmentation' : False,
            'remove_far_background': False,
            'remove_far_background_all' : False,
            'no_background' : False,
            'height_map_min_value' : -0.05,
            'height_map_value_range' : 0.25,
            'height_diff_clip_range' : [-0.1,0.1],
            'weights_prefix_to_load' : [],
            'far_away_threshold' : 1.4,
            'use_predict_precondition' : False,
            'precondition_input_dim' : 1,
            'precondition_threshold' : 0.2,
            'precondition_load' : '',
        }
        return hp
    
    # def act(self, t=None, i_tr=None, state=None):
    #     return super(ForwardModelController, self).act(t, i_tr, state)
    
    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        '''
        :params processed_actions: (num_samples, nactions, 5)
        '''
        self.predictor.eval()
        num_samples, nactions, _ = processed_actions.shape
        depth_image = self._state["depth"]
        height_image = self._state["height"]
        color_image = self._state["rgb"]
        invalid_depth_mask = np.isclose(depth_image, 0.0, rtol=0.0, atol=1e-4) # coordinates where height estimation is not
        invalid_height_mask = height_image < -0.05
        invalid_mask = np.logical_or(invalid_depth_mask, invalid_height_mask)
        if self._hp['remove_far_background_all']:
            far_depth_mask = np.logical_or(depth_image > self._hp['far_away_threshold'], invalid_depth_mask)
            
            far_depth_mask_transformed = cv2.warpPerspective(np.asarray(far_depth_mask).astype(np.float32), \
                                                    self.homography_matrix, (self.space_revealed_map_w, self.space_revealed_map_h), flags=cv2.INTER_LINEAR)
        color_heightmap, height_map = plant_utils.homography_transform_color_and_depth_image(color_image, height_image, self.homography_matrix, self.space_revealed_map_w, self.space_revealed_map_h)
        height_map_mask= cv2.warpPerspective(np.asarray(invalid_mask).astype(np.float32), self.homography_matrix, (self.space_revealed_map_w, self.space_revealed_map_h), flags=cv2.INTER_LINEAR)
        height_map_mask[height_map_mask > 0] = 1.0
        height_map_mask = height_map_mask.astype(int) #.astype(bool)
        if self._hp['remove_far_background_all']:
            far_depth_mask_transformed[far_depth_mask_transformed > 0] = 1.0
            far_depth_mask_transformed = far_depth_mask_transformed.astype(bool)
            height_map_near_ground = np.logical_or(np.isclose(height_map, 0, rtol=0.0, atol=0.05), height_map<0)
            far_depth_mask_transformed = np.logical_or(far_depth_mask_transformed, height_map_near_ground)
            far_depth_mask_transformed = far_depth_mask_transformed.astype(int)

        processed_actions_flatten = processed_actions.reshape((num_samples * nactions, -1)) 
        x_mins = processed_actions_flatten[:,0] - self._hp["img_input_distance_to_boundary"]
        y_maxs = processed_actions_flatten[:,1] + self._hp["img_input_distance_to_boundary"]
        all_workspace_limits_center_around_action = np.hstack([x_mins.reshape(-1,1), y_maxs.reshape(-1,1)])
        bound_coords_2d_x2, bound_coords_2d_y2 = plant_utils.get_coordinate_in_image(all_workspace_limits_center_around_action, self.workspace_for_reveal[0][0], self.workspace_for_reveal[1][0], self.space_revealed_map_w, self.space_revealed_map_h, self.img_input_resolution, restrict_to_within_image=True) # , pixel_include_upper=False
        bound_coords_2d = np.hstack([
            np.zeros(len(bound_coords_2d_x2)).reshape(-1,1),
            bound_coords_2d_x2.reshape(-1,1),
            bound_coords_2d_y2.reshape(-1,1),
            (bound_coords_2d_x2 + self.img_input_w-1).reshape(-1,1),
            (bound_coords_2d_y2 + self.img_input_h-1).reshape(-1,1),
        ])
        bound_coords_2d_tensor = torch.FloatTensor(bound_coords_2d)
        # import pdb;pdb.set_trace()
        # color height map patches (num_samples * nactions, 3, patch_w, patch_h)
        # height map patches (num_samples * nactions, 1, patch_w, patch_h)
        # mask patches (num_samples * nactions, 1, patch_w, patch_h)
        color_heightmap_tensor = torch.FloatTensor(np.asarray(color_heightmap).transpose((2, 0, 1))).unsqueeze(0) # (1, 3, h, w)
        height_map_tensor = torch.FloatTensor(height_map[np.newaxis, ...]).unsqueeze(0) # (1, 1, h, w)
        height_map_mask_tensor = torch.FloatTensor(height_map_mask[np.newaxis, ...]).unsqueeze(0)
        

        color_heightmap_patches = torchvision.ops.roi_pool(color_heightmap_tensor, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)
        height_map_patches = torchvision.ops.roi_pool(height_map_tensor, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)
        height_map_mask_patches = torchvision.ops.roi_pool(height_map_mask_tensor, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)
        if self._hp['remove_far_background_all']:
            far_depth_mask_tensor = torch.FloatTensor(far_depth_mask_transformed[np.newaxis, ...]).unsqueeze(0)
            far_depth_patches = torchvision.ops.roi_pool(far_depth_mask_tensor, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)
        
        # action_info: first num_angles dimensions are angles
        thetas = processed_actions_flatten[:,3]
        lengths = processed_actions_flatten[:,4]
        angle_indices = np.floor((thetas - self.theta_sampling_config['min']) / self.theta_sampling_config['discrete']).astype(int)
        angle_one_hot = F.one_hot(torch.tensor(angle_indices).long(), self.num_angles) # (num_samples * nactions, 7)
        # action_info: last 2 dimensions 
        x_ends, y_ends = calculate_action_end(thetas, lengths, self._hp["distance_to_boundary"])
        action_end = torch.FloatTensor(np.stack([x_ends,y_ends]).T)
        all_action_info = torch.cat([angle_one_hot, action_end], dim=1) # (num_samples * nactions, 9)
        # import pdb;pdb.set_trace()
        

        # plant_dataset = PlantInferenceDataset(color_heightmap_patches, height_map_patches, height_map_mask_patches, all_action_info)
        # loader_args = dict(batch_size=self._hp['batch_size'] * self._hp["nactions"], num_workers=6, pin_memory=True)
        # plant_loader = DataLoader(plant_dataset, shuffle=False, drop_last=False, **loader_args)
        all_scores = []
        all_new_space_revealed = []
        all_intermediate_space_revealed = []
        all_probs = []
        all_thresholded = []
        sample_idx_tally=0
        # fold_indices = (actions[:,:,0] + actions[:,:,1] * self.num_local_patches_width).astype(int) # (num_samples, nactions)
        
        batchsize = self._hp['batch_size'] * self._hp["nactions"]
        num_batches = len(color_heightmap_patches) // batchsize  # num_samples * nactions
        if num_batches * batchsize < len(color_heightmap_patches):
            num_batches += 1
        # for batch_idx, batch in enumerate(plant_loader): # in range(num_batches):#, 
        height_map_patches = normalize_height_map(height_map_patches, min_value=self._hp['height_map_min_value'], value_range=self._hp['height_map_value_range']) 
        ind1,ind2,ind3,ind4 = torch.where(height_map_mask_patches)
        height_map_patches[ind1,ind2,ind3,ind4] = 0
        if self._hp['remove_far_background_all']:
            for color_ch in range(3):
                # import pdb;pdb.set_trace()
                color_heightmap_patches[:,color_ch:color_ch+1,:,:][torch.where(far_depth_patches)] = 0
           
        color_heightmap_patches = color_heightmap_patches / 255
        for batch_idx in range(num_batches):
           
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx+1) * batchsize, len(color_heightmap_patches))
            if self._hp['use_height_feature']:
                image = torch.cat([color_heightmap_patches[start_idx:end_idx], height_map_patches[start_idx:end_idx]],dim=1) 
            else:
                image = color_heightmap_patches[start_idx:end_idx]
            action_info = all_action_info[start_idx:end_idx]
            image_loss_mask = 1 - height_map_mask_patches[start_idx:end_idx][:,0,...].float()
            
            image = image.to(device=self.device, dtype=torch.float32)
            action_info = action_info.to(device=self.device, dtype=torch.float32)
            image_loss_mask = image_loss_mask.to(device=self.device, dtype=torch.float32)
            # print(image.shape, action_info.shape, image_loss_mask.shape)
            with torch.no_grad():
                mask_pred = self.predictor(image, action_info) # (B, num_ch, h, w)
                if self._hp['use_regression_loss']:
                    masks_pred_discrete = mask_pred[:,:-1,:,:]
                    masks_pred_cont = mask_pred[:,-1,:,:]
                else:
                    masks_pred_discrete = mask_pred
                if self._hp['use_predict_precondition']:
                    precond_pred = self.precond_predictor(image, action_info)
                    precond_satisfy = precond_pred[:,1] >= self._hp['precondition_threshold']
                mask_pred_softmax = F.softmax(masks_pred_discrete, dim=1).float()
                mask_pred_softmax = mask_pred_softmax * image_loss_mask.unsqueeze(1)
                height_decrease_class_prob  = torch.sum(mask_pred_softmax[:,self._hp['threshold_class']:,...], dim=1) # (B, h, w)
                # import pdb;pdb.set_trace()
                height_decrease_thresholded  = metrics_utils.threshold_prob(height_decrease_class_prob, threshold=self._hp['iou_threshold']) # (B, h, w)
                # import pdb;pdb.set_trace()
                all_probs.append(height_decrease_class_prob.detach().cpu().numpy())
                all_thresholded.append(height_decrease_thresholded.detach().cpu().numpy())

                for sample_idx in range(len(image) // self._hp["nactions"]):
                    current_space_revealed_maps = []
                    current_space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
                    for step_idx in range(self._hp["nactions"]):
                        b_idx = sample_idx * self._hp["nactions"] + step_idx
                        space_revealed_by_action_i = height_decrease_thresholded[b_idx].detach().cpu().numpy() # (h,w)
                        if self._hp['use_predict_precondition']:
                            if not precond_satisfy[b_idx].item():
                                current_space_revealed_maps.append(np.copy(current_space_revealed_map))
                                continue
                        xpixel= bound_coords_2d_x2[sample_idx_tally] #[b_idx][0], bound_coords_2d_y[b_idx][1]
                        ypixel = bound_coords_2d_y2[sample_idx_tally]
                        old_patch = current_space_revealed_map[ypixel:ypixel+self.img_input_h, xpixel:xpixel+self.img_input_w]
                        current_space_revealed_map[ypixel:ypixel+self.img_input_h, xpixel:xpixel+self.img_input_w] = np.maximum(space_revealed_by_action_i, old_patch)

                        current_space_revealed_maps.append(np.copy(current_space_revealed_map))
                        
                        sample_idx_tally+=1
                    all_intermediate_space_revealed.append(current_space_revealed_maps)
                    all_new_space_revealed.append(current_space_revealed_map)
                    new_discovery = current_space_revealed_map - self.space_revealed_map
                    new_discovery = new_discovery[self.y_start_calculate:self.y_end_calculate+1, self.x_start_calculate:self.x_end_calculate+1]
                    score = new_discovery[new_discovery > 0].sum()
                    all_scores.append(score)

        self._new_space_revealed_map_after_all_actions = np.stack(all_new_space_revealed)
        self._all_probs = np.vstack(all_probs).reshape((-1, self._hp["nactions"], self.img_input_h, self.img_input_w))
        self._all_thresholded = np.vstack(all_thresholded).reshape((-1, self._hp["nactions"], self.img_input_h, self.img_input_w))
        self._all_intermediate_space_revealed = all_intermediate_space_revealed
        self._reveal_map_xy_start = bound_coords_2d[:,1:3].astype(int).reshape((-1, self._hp["nactions"],2))
        self._height_maps = [color_heightmap_patches, height_map_patches, height_map_mask_patches]
        self._all_action_info = all_action_info
        return np.asarray(all_scores)
    
    def update_reveal_info(self, afteraction_height_map_mask, height_map_diff_original, new_space_revealed_map_gt_original, pi_t):
        if not self._hp['adaptive_train']:
            return
        self.image_tensor_acc.append(pi_t['rgb_image_input'][0]) #(3,h,w)
        self.height_tensor_acc.append(pi_t['height_image_input'][0].unsqueeze(0)) #(1,h,w), already masked by the initial frame mask
        before_img_mask = pi_t['img_mask_input'][0]
        after_img_mask = torch.as_tensor(afteraction_height_map_mask.astype(int))
        # img_mask = torch.amax(torch.stack([before_img_mask,after_img_mask]),dim=0)
        self.img_mask_acc.append(before_img_mask) # (h,w)
        self.action_info_acc.append(pi_t['action_info_input'][0]) # (9,)
        xpixel, ypixel = pi_t['reveal_map_xy_start'][0]
        mask_gt = new_space_revealed_map_gt_original[ypixel:ypixel+self.img_input_h, xpixel:xpixel+self.img_input_w]
        self.gt_mask_acc.append(torch.FloatTensor(mask_gt).contiguous()) # (h,w)
        # self.height_diff_acc.append(height_map_diff_original)
        self.postaction_mask_acc.append(after_img_mask.unsqueeze(0))
        
    
    def adaptive_train(self):
        self.predictor.load_state_dict(torch.load(self._hp['load'], map_location=self.device))
        self.predictor.train()
        images = torch.stack(self.image_tensor_acc, axis=0) # (n, 3, h, w)
        true_masks = torch.stack(self.img_mask_acc, axis=0) # (n, 1, h, w)
        images = images.to(device=self.device, dtype=torch.float32)
        true_masks = true_masks.to(device=self.device, dtype=torch.long)
        # action_info = action_info.to(device=self.device, dtype=torch.float32)
        # image_loss_mask = image_loss_mask.to(device=self.device, dtype=torch.float32)
        masks_pred = self.predictor(images, action_info)

        if self._hp['use_regression_loss']:
            masks_pred_discrete = masks_pred[:,:-1,:,:]
            masks_pred_cont = masks_pred[:,-1,:,:]
        else:
            masks_pred_discrete = masks_pred
        if self._hp['use_postaction_mask_loss']:
            masks_pred_postaction_mask = masks_pred_discrete[:,-2:,:,:]
            masks_pred_discrete = masks_pred_discrete[:,:-2,:,:]
        
        loss_all = self.criterion(masks_pred_discrete, true_masks) * image_loss_mask 
        loss = loss_all.sum() / image_loss_mask.sum() 
        if len(self._hp['classes']) > 1:
            mask_true_one_hot = F.one_hot(true_masks, len(self._hp['classes'])+1).permute(0, 3, 1, 2)
            mask_pred_softmax = F.softmax(masks_pred_discrete, dim=1)
            background_class_lprob = torch.log(torch.sum(mask_pred_softmax[:,:self._hp['threshold_class'],...], dim=1))
            height_decrease_class_lprob  = torch.log(torch.sum(mask_pred_softmax[:,self._hp['threshold_class']:,...], dim=1))
            true_masks_2class = torch.clone(true_masks)
            true_masks_2class[true_masks_2class < self._hp['threshold_class']] = 0
            true_masks_2class[true_masks_2class >= self._hp['threshold_class']] = 1
            loss_background_foreground_all = self.criterion1(torch.stack([background_class_lprob, height_decrease_class_lprob], dim=1), true_masks_2class) * image_loss_mask
            loss_background_foreground = loss_background_foreground_all.sum() / image_loss_mask.sum() 
            loss += loss_background_foreground

        if self._hp['use_regression_loss']:
            # height_diff = batch['height_diff'].to(device=device, dtype=torch.float32)
            reg_loss = self.criterion2(masks_pred_cont, height_diff) * image_loss_mask 
            reg_loss = reg_loss.sum() / image_loss_mask.sum() 
            loss += reg_loss
        if self._hp['use_postaction_mask_loss']:
            # postaction_mask = postaction_mask.to(device=self.device, dtype=torch.long)
            postaction_mask_loss_all = self.criterion(masks_pred_postaction_mask, postaction_mask) * image_loss_mask 
            postaction_mask_loss = postaction_mask_loss_all.sum() / image_loss_mask.sum() 
            loss += postaction_mask_loss

        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


    def act(self, t=None, i_tr=None, state=None):
        if t > 0 and self._hp['adaptive_train']:
            self.adaptive_train()
        pi_t = super(ForwardModelController, self).act(t, i_tr, state)
        if self._hp['exhaustive']:
            return pi_t
        idx_in_dataset = self._best_indices[0] * self._hp['nactions']
        additional_info = {
            'reveal_map_xy_start' : self._reveal_map_xy_start[self._best_indices[0]],
            "pred" : self._all_thresholded[self._best_indices[0]], 
            "pred_prob" : self._all_probs[self._best_indices[0]], 
            "rgb_image_input" : self._height_maps[0][idx_in_dataset:idx_in_dataset+self._hp['nactions']], 
            "height_image_input" : self._height_maps[1][idx_in_dataset:idx_in_dataset+self._hp['nactions']][:,0,...],
            "img_mask_input" : self._height_maps[2][idx_in_dataset:idx_in_dataset+self._hp['nactions']][:,0,...],
            "action_info_input" : self._all_action_info[idx_in_dataset:idx_in_dataset+self._hp['nactions']],
        }
        
        pi_t['additional_info'] = additional_info
        return pi_t


class ForwardModelCameraFrameController(CEMBaseController):
    def __init__(self, policyparams, envparams):
        CEMBaseController.__init__(self, policyparams, envparams)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theta_sampling_config = self._hp['initial_sampling_configurations']['theta']
        
        self.img_input_w = self.img_input_h = self._hp['input_image_size']
        self.crop_size = self._hp['crop_size']

        self.num_angles = self._hp['num_angles']
        z_level_config = self._hp['initial_sampling_configurations']['z_level']
        self.num_z_levels = int((z_level_config['max'] - z_level_config['min']) / z_level_config['discrete']) + 1

        self._hp['predict_precondition'] = False
        self.predictor = UNet(self._hp)
        self.predictor.load_state_dict(torch.load(self._hp['load'], map_location=self.device))
        self.predictor.to(device=self.device)

        # transformations
        
    
    # def project_pts_to_img(self, pts):
    #     pts = PointCloud(np.asarray(pts).T, frame="world")
    #     pts_camera = self.T_camera_world.inverse() * pts
    #     x_coords, y_coords = self.intrinsic_iam.project(pts_camera).data
    #     return x_coords, y_coords
    
    def sampled_actions_in_world_frame(self, pts):
        pts = PointCloud(np.asarray(pts[:,:3]).T, frame="azure_kinect_overhead_upright")
        pts_world = self.T_sampled_action_space_to_world * pts
        
        pts_world = pts_world.data.T #(nsamples, 3)
        pts_world[:,-1] += 0.26
        pts_world[:,-1] -= 0.35-0.107
        return pts_world

    def reset(self):
        super(ForwardModelCameraFrameController, self).reset()

    def _additional_params(self):
        hp = {
            'is_forward_model' : True,
            'adaptive_train' : False,
            'num_angles' : 8,
            'crop_size' : 300,
            'minimum_decrease_before_transformation' : -0.05,
            'input_image_size' : 200,
            'distance_to_boundary' : 0.15,
            'iou_threshold' : 0.5,
            'use_height_feature' : True,
            'no_rgb_feature' : False,
            'action_as_image' : False,
            'no_image' : False,
            'no_action' : False,
            'use_coordinate_feature' : False, # TODO
            'classes' : [-0.05],
            'threshold_class' : 1,
            'huber_loss_delta' : 0.1,
            'use_regression_loss' : False,
            'use_postaction_mask_loss' : False,
            'upconv_channels' : [512, 512, 256, 256, 256, 256, 128, 128, 64, 64],
            'use_resnet' : False,
            'resnet_maxpool_after_enc' : False,
            'downconv_channels' : [64, 64, 128, 256, 512, 512],
            'unet_downsample_layer_type' : 'DownMaxpoolSingleConv',
            'action_upconv_channels' : [64, 64, 64, 128, 128],
            'action_upconv_sizes' : [3,6],
            'action_input_dim' : 13,
            'load' : "",
            'batch_size' : 10,
            'no_background' : False,
            'height_map_min_value' : 0.0,
            'height_map_value_range' : 1.0,
            'height_diff_clip_range' : [-1.4, 0.0],
            'weights_prefix_to_load' : [],
            'far_away_threshold' : 1.4,
            'predict_precondition' : False,
            'use_predict_precondition' : False,
        }
        return hp
    

    def evaluate_rollouts(self, actions, processed_actions, cem_itr):
        '''
        :params processed_actions: (num_samples, nactions, 5)
        '''
        self.predictor.eval()
        num_samples, nactions, _ = processed_actions.shape
        
        height_image = self._state["height_f"]
        color_image = self._state["rgb"]

        processed_actions_flatten = processed_actions.reshape((num_samples * nactions, -1)) 
        proc_actions_world = self.sampled_actions_in_world_frame(processed_actions_flatten)
        # import pdb;pdb.set_trace()
        action_center_xs, action_center_ys = self.project_pts_to_img(proc_actions_world)
        
        # start_time = time.time()
        bound_coords_2d_x2 = action_center_xs - self.crop_size // 2
        bound_coords_2d_y2 = action_center_ys - self.crop_size // 2
        color_heightmap_patches = []
        height_map_patches = []
        for start_pixel_x, start_pixel_y in zip(bound_coords_2d_x2, bound_coords_2d_y2):
            height_img = np.copy(height_image)[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
            color_img = np.copy(color_image)[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size,:]
            height_img = normalize_height_map(height_img, min_value=self._hp['height_map_min_value'], value_range=self._hp['height_map_value_range']) 

            height_img = cv2.resize(height_img, (self.img_input_h, self.img_input_w))
            color_img = cv2.resize(color_img, (self.img_input_h, self.img_input_w))

            color_heightmap_patches.append(torch.FloatTensor(np.asarray(color_img).transpose((2, 0, 1))))
            height_map_patches.append(torch.FloatTensor(height_img[np.newaxis, ...]))
        
        color_heightmap_patches = torch.stack(color_heightmap_patches)
        height_map_patches = torch.stack(height_map_patches)
        # end_time = time.time()
        # time_taken = end_time - start_time
        # print("Evaluate Time taken: {:.3f} seconds".format(time_taken))

        
        
        bound_coords_2d = np.hstack([
            np.zeros(len(bound_coords_2d_x2)).reshape(-1,1),
            bound_coords_2d_x2.reshape(-1,1),
            bound_coords_2d_y2.reshape(-1,1),
            (bound_coords_2d_x2 + self.crop_size-1).reshape(-1,1),
            (bound_coords_2d_y2 + self.crop_size-1).reshape(-1,1),
        ])
        # bound_coords_2d_tensor = torch.FloatTensor(bound_coords_2d)
        
        # color_heightmap_tensor2 = torch.FloatTensor(np.asarray(color_image).transpose((2, 0, 1))).unsqueeze(0) # (1, 3, h, w)
        # height_map_tensor2 = torch.FloatTensor(height_image[np.newaxis, ...]).unsqueeze(0) # (1, 1, h, w)
        

        # color_heightmap_patches2 = torchvision.ops.roi_pool(color_heightmap_tensor2, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)
        # height_map_patches2 = torchvision.ops.roi_pool(height_map_tensor2, bound_coords_2d_tensor, (self.img_input_w,self.img_input_h), 1.0)

        
        
        # action_info: first num_angles dimensions are angles
        thetas = processed_actions_flatten[:,3]
        lengths = processed_actions_flatten[:,4]
        z_levels = processed_actions_flatten[:,5].astype(int)
        angle_indices = np.floor((thetas - self.theta_sampling_config['min']) / self.theta_sampling_config['discrete']).astype(int)
        angle_one_hot = F.one_hot(torch.tensor(angle_indices).long(), self.num_angles) # (num_samples * nactions, num_angles)
        z_level_one_hot = F.one_hot(torch.tensor(z_levels).long(), self.num_z_levels)
        # action_info: last 2 dimensions 
        x_ends, y_ends = calculate_action_end(thetas, lengths, self._hp["distance_to_boundary"])
        action_end = torch.FloatTensor(np.stack([x_ends,y_ends]).T)
        all_action_info = torch.cat([angle_one_hot, z_level_one_hot, action_end], dim=1) # (num_samples * nactions, 13)
        assert all_action_info.shape[1] == self._hp['action_input_dim']
        # import pdb;pdb.set_trace()

        
        # fold_indices = (actions[:,:,0] + actions[:,:,1] * self.num_local_patches_width).astype(int) # (num_samples, nactions)
        
        batchsize = self._hp['batch_size'] * self._hp["nactions"]
        num_batches = len(color_heightmap_patches) // batchsize  # num_samples * nactions
        if num_batches * batchsize < len(color_heightmap_patches):
            num_batches += 1
        
        # height_map_patches = normalize_height_map(height_map_patches, min_value=self._hp['height_map_min_value'], value_range=self._hp['height_map_value_range']) 
        color_heightmap_patches = color_heightmap_patches / 255

        up_sampling = nn.Upsample(scale_factor=self.crop_size / self.img_input_h, mode='bilinear')
        
        all_scores = []
        all_new_space_revealed = []
        all_intermediate_space_revealed = []
        all_probs = []
        all_thresholded = []
        sample_idx_tally=0

        # batch_start_time = time.time()
        for batch_idx in range(num_batches):
           
            start_idx = batch_idx * batchsize
            end_idx = min((batch_idx+1) * batchsize, len(color_heightmap_patches))
            if self._hp['use_height_feature']:
                image = torch.cat([color_heightmap_patches[start_idx:end_idx], height_map_patches[start_idx:end_idx]],dim=1) 
            else:
                image = color_heightmap_patches[start_idx:end_idx]
            action_info = all_action_info[start_idx:end_idx]
            
            
            image = image.to(device=self.device, dtype=torch.float32)
            action_info = action_info.to(device=self.device, dtype=torch.float32)
            # image_loss_mask = image_loss_mask.to(device=self.device, dtype=torch.float32)
            # print(image.shape, action_info.shape, image_loss_mask.shape)
            with torch.no_grad():
                # start_time = time.time()
                mask_pred = self.predictor(image, action_info) # (B, num_ch, h, w)
                if self._hp['use_regression_loss']:
                    masks_pred_discrete = mask_pred[:,:-1,:,:]
                    masks_pred_cont = mask_pred[:,-1,:,:]
                else:
                    masks_pred_discrete = mask_pred
                
                mask_pred_softmax = F.softmax(masks_pred_discrete, dim=1).float()
                # mask_pred_softmax = mask_pred_softmax * image_loss_mask.unsqueeze(1)
                height_decrease_class_prob  = torch.sum(mask_pred_softmax[:,self._hp['threshold_class']:,...], dim=1) # (B, h, w)
                # import pdb;pdb.set_trace()
                height_decrease_class_prob_upsampled = up_sampling(height_decrease_class_prob.unsqueeze(1)).squeeze(1)
                # end_time = time.time()
                # time_taken = end_time - start_time
                # print("Calculate model output Time taken: {:.3f} seconds".format(time_taken))

                        
                height_decrease_thresholded  = metrics_utils.threshold_prob(height_decrease_class_prob_upsampled, threshold=self._hp['iou_threshold']) # (B, h, w)
                
                all_probs.append(height_decrease_class_prob_upsampled.detach().cpu().numpy())
                all_thresholded.append(height_decrease_thresholded.detach().cpu().numpy())

                # start_time = time.time()
                for sample_idx in range(len(image) // self._hp["nactions"]):
                    current_space_revealed_maps = []
                    current_space_revealed_map = np.zeros((self.space_revealed_map_h, self.space_revealed_map_w))
                    for step_idx in range(self._hp["nactions"]):
                        b_idx = sample_idx * self._hp["nactions"] + step_idx
                        space_revealed_by_action_i = height_decrease_thresholded[b_idx].detach().cpu().numpy() # (h,w)
                        
                        xpixel= bound_coords_2d_x2[sample_idx_tally] 
                        ypixel = bound_coords_2d_y2[sample_idx_tally]
                        old_patch = current_space_revealed_map[ypixel:ypixel+self.crop_size, xpixel:xpixel+self.crop_size]
                        current_space_revealed_map[ypixel:ypixel+self.crop_size, xpixel:xpixel+self.crop_size] = np.maximum(space_revealed_by_action_i, old_patch)

                        current_space_revealed_maps.append(np.copy(current_space_revealed_map))
                        
                        sample_idx_tally+=1
                    # all_intermediate_space_revealed: (a list) what space does each action reveal?
                    all_intermediate_space_revealed.append(current_space_revealed_maps)
                    # all_new_space_revealed: collectively what space is revealed?
                    all_new_space_revealed.append(current_space_revealed_map)
                    new_discovery = current_space_revealed_map - self.space_revealed_map
                    new_discovery = new_discovery[self.y_start_calculate:self.y_end_calculate+1, self.x_start_calculate:self.x_end_calculate+1]
                    score = new_discovery[new_discovery > 0].sum()
                    all_scores.append(score)
                # end_time = time.time()
                # time_taken = end_time - start_time
                # print("Calculate space revealed Evaluate Time taken: {:.3f} seconds".format(time_taken))
        
        # batch_end_time = time.time()
        # time_taken = batch_end_time - batch_start_time
        # print("All batches Evaluate Time taken: {:.3f} seconds".format(time_taken))
        
        self._new_space_revealed_map_after_all_actions = np.stack(all_new_space_revealed)
        self._all_probs = np.vstack(all_probs).reshape((-1, self._hp["nactions"], self.crop_size, self.crop_size))
        self._all_thresholded = np.vstack(all_thresholded).reshape((-1, self._hp["nactions"], self.crop_size, self.crop_size))
        self._all_intermediate_space_revealed = all_intermediate_space_revealed
        self._reveal_map_xy_start = bound_coords_2d[:,1:3].astype(int).reshape((-1, self._hp["nactions"],2))
        self._height_maps = [color_heightmap_patches, height_map_patches]
        self._all_action_info = all_action_info
        return np.asarray(all_scores)

    def act(self, t=None, i_tr=None, state=None):
        if t > 0 and self._hp['adaptive_train']:
            self.adaptive_train()
        pi_t = super(ForwardModelCameraFrameController, self).act(t, i_tr, state)
        if self._hp['exhaustive']:
            return pi_t
        idx_in_dataset = self._best_indices[0] * self._hp['nactions']
        additional_info = {
            'reveal_map_xy_start' : self._reveal_map_xy_start[self._best_indices[0]],
            "pred" : self._all_thresholded[self._best_indices[0]], 
            "pred_prob" : self._all_probs[self._best_indices[0]], 
            "rgb_image_input" : self._height_maps[0][idx_in_dataset:idx_in_dataset+self._hp['nactions']], 
            "height_image_input" : self._height_maps[1][idx_in_dataset:idx_in_dataset+self._hp['nactions']][:,0,...],
            "action_info_input" : self._all_action_info[idx_in_dataset:idx_in_dataset+self._hp['nactions']],
        }
        
        pi_t['additional_info'] = additional_info
        return pi_t