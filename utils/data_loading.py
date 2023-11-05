import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import pickle
from PIL import Image
import cv2
import utils.utils as unet_utils
import utils.plant_utils as plant_utils
from tqdm import tqdm

def calculate_action_end(thetas, lengths, distance_to_boundary):
    x_end = (np.cos(thetas) * lengths + distance_to_boundary) / (distance_to_boundary * 2)
    y_end = (np.sin(thetas) * lengths + distance_to_boundary) / (distance_to_boundary * 2)
    return x_end, y_end

def normalize_height_map(height_img, min_value = -0.05, value_range = 0.25):
    height_img = np.clip(height_img, min_value, min_value+value_range)
    return (height_img - min_value) / value_range

def generate_mask(height_diff, height_diff_threshold):
    height,width = height_diff.shape[:2]
    mask = np.ones((height, width)) #.astype(int)
    mask[height_diff > height_diff_threshold[0]] = 0
    for height_type in range(len(height_diff_threshold)):
        if height_type == 0:
            continue
        upper_height_diff = height_diff_threshold[height_type-1]
        lower_height_diff = height_diff_threshold[height_type]
        
        mask[(height_diff <= upper_height_diff) & (height_diff > lower_height_diff)] = height_type  
    mask[height_diff <= height_diff_threshold[-1]] = len(height_diff_threshold)
    
    return mask

def generate_icp_aligned_mask(diff, diff_real, height_diff_threshold, minimum_decrease_before_transformation):
    # mask = diff <= minimum_decrease
    # # mask = cv2.medianBlur(mask.astype(np.uint8), 5).astype(bool)
    # mask_new = np.logical_and(mask, diff_real <= minimum_decrease_before_transformation)
    # mask_new = cv2.medianBlur(mask_new.astype(np.uint8), 5).astype(bool) #np.logical_or(mask, mask_new)
    
    def valid_decrease(maski):
        return np.logical_and(maski, diff_real <= minimum_decrease_before_transformation)

    height,width = diff.shape[:2]
    mask = np.zeros((height, width)) #.astype(int)
    # mask[diff > height_diff_threshold[0]] = 0
    for height_type in range(len(height_diff_threshold)):
        if height_type == 0:
            continue
        upper_height_diff = height_diff_threshold[height_type-1]
        lower_height_diff = height_diff_threshold[height_type]
        if upper_height_diff < 0 and lower_height_diff < 0:
            this_type_mask = valid_decrease((diff <= upper_height_diff) & (diff > lower_height_diff))
        else:
            this_type_mask = (diff <= upper_height_diff) & (diff > lower_height_diff)
        mask[this_type_mask] = height_type  
    mask[valid_decrease(diff <= height_diff_threshold[-1])] = len(height_diff_threshold)
    
    return mask

class PlantDatasetCameraFrame(Dataset):
    def __init__(self, df, split, hp, num_angles=8):
        self.split = split
        self.df = df
        self.height_diff_threshold = hp['classes']
        self.num_angles = 8

        self.no_gt = False

        self.hp = hp
        self.use_regression_loss = hp['use_regression_loss']
        self.use_height_feature = hp['use_height_feature']
        self.height_diff_min, self.height_diff_max = hp['height_diff_clip_range']

        color_jitter_b, color_jitter_c, color_jitter_s, color_jitter_h = hp['color_jitter_setting']
        self.color_jitter_aug = torchvision.transforms.ColorJitter(brightness=color_jitter_b, contrast=color_jitter_c, saturation=color_jitter_s, hue=color_jitter_h)
        
        self.crop_size = hp['crop_size']
        self.img_height = self.img_width = hp['input_image_size']

        if 'data_key' in self.df:
            data_keys = self.df['data_key'].unique()
            all_indices = []
            
            for key in data_keys:
                indices = df.index[(df['data_key'] == key)].tolist()
                all_indices.append(indices)
        else:
            all_indices = [[idx] for idx in range(len(df))]
        self.all_indices = all_indices
        self.negative_sample_idx = []
        self.positive_sample_idx = []
        self.pre_load()

        self.negative_sample_idx = list(set(self.negative_sample_idx))
        self.positive_sample_idx = list(set(self.positive_sample_idx))
        print(f":: #self.positive_sample_idx={len(self.positive_sample_idx)} #negative samples={len(self.negative_sample_idx)}")
        # if split == "train":
        #     self.data_set_length = len(self.positive_sample_idx) * 2
        #     self.reset()
        # else:
        self.data_set_length = len(self.all_indices)


    def pre_load(self):
        all_processed_data = []
        for data_idx in tqdm(range(len(self.all_indices))):
            processed_data = []
            for idx in self.all_indices[data_idx]:
                entry = self.df.iloc[idx]
                pt_img_x, pt_img_y = entry['pt_img_x'], entry['pt_img_y']
                pte_img_x, pte_img_y = entry['pte_img_x'], entry['pte_img_y']
                start_pixel_x = pt_img_x - self.crop_size // 2
                start_pixel_y = pt_img_y - self.crop_size // 2

                rgb = cv2.cvtColor(cv2.imread(entry['initial_color_path']), cv2.COLOR_BGR2RGB)    
                height1 = plant_utils.read_depth_image(entry['initial_height_map_path_blurred'])
                height_img = height1[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
                color_img = rgb[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size,:]
                
                try:
                    height2 = plant_utils.read_depth_image(entry['after_action_height_map_path'])[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
                    invalid_after = height2 == 0
                except:
                    invalid_after = np.zeros(height1.shape).astype(bool)
                
                try:
                    height3 = plant_utils.read_depth_image(entry['after_action_original_height_map_path'])[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
                    invalid_after_original = height3 == 0
                except:
                    invalid_after_original = np.zeros(height1.shape).astype(bool)

                img_mask = np.logical_or(invalid_after_original, invalid_after) #np.logical_or(, invalid_height_before)
                if self.img_height < self.crop_size:
                    img_mask_smaller = cv2.resize(img_mask.astype(np.float64), (self.img_height, self.img_width))
                    img_mask_smaller[img_mask_smaller < 1] = 0
                    img_mask_smaller = img_mask_smaller.astype(int)


                diff = np.load(entry["diff_path"])[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
                diff_real = np.load(entry["diff_real_path"])[start_pixel_y:start_pixel_y+self.crop_size,start_pixel_x:start_pixel_x+self.crop_size]
                # blurring?
                diff = cv2.medianBlur(diff.astype(np.float32),5).astype(np.float64)
                diff_real = cv2.medianBlur(diff_real.astype(np.float32),5).astype(np.float64)
                decrease_that_counts = diff_real <= self.hp['minimum_decrease_before_transformation']
                mask = generate_icp_aligned_mask(diff, diff_real, self.height_diff_threshold, self.hp['minimum_decrease_before_transformation'])
                
                height_img = normalize_height_map(height_img, min_value=self.hp['height_map_min_value'], value_range=self.hp['height_map_value_range'])
                
                height_diff = np.copy(diff)
                height_diff[np.where(np.logical_and(height_diff < 0, np.logical_not(decrease_that_counts)))] = 0.0
                height_diff = np.clip(height_diff, self.height_diff_min, self.height_diff_max)
                height_diff = (height_diff - self.height_diff_min) / (self.height_diff_max - self.height_diff_min)
                # height_diff = height_diff * 100

                if self.img_height < self.crop_size:
                    height_img = cv2.resize(height_img, (self.img_height, self.img_width))
                    color_img = cv2.resize(color_img, (self.img_height, self.img_width))
                    mask = cv2.resize(mask, (self.img_height, self.img_width)).astype(int)
                    height_diff = cv2.resize(height_diff, (self.img_height, self.img_width))
                
                # percentage of revealed pixel
                perc_pixel_revealed = np.sum(mask != 0) / (self.img_height * self.img_width)
                if perc_pixel_revealed < 0.05:
                    self.negative_sample_idx.append(data_idx)
                else:
                    self.positive_sample_idx.append(data_idx)

                data = {
                    "depth_img" : height_img, 
                    "color_img" :  color_img, 
                    "img_mask" : img_mask_smaller,#np.zeros((self.img_height, self.img_width)).astype(bool), 
                    "img_mask_smaller" : img_mask_smaller,
                    "mask" : mask, 
                    'original_height_diff' : height_diff,
                    'pix_x' : 0.5,
                    'pix_y' : 0.5,
                    'pix_x_dx' : (pte_img_x - start_pixel_x) / float(self.crop_size),
                    'pix_y_dy' : (pte_img_y - start_pixel_y) / float(self.crop_size),
                    'perc_pixel_revealed' : perc_pixel_revealed,
                }
                data['pix_x_dx'] = min(max(0.0, data['pix_x_dx']), 1.0)
                data['pix_y_dy'] = min(max(0.0, data['pix_y_dy']), 1.0)
                # if data['pix_x_dx'] < 0 or data['pix_x_dx'] >= self.img_width:
                #     print(":: Error! pix_x_dx exceeds image size")
                #     continue 
                processed_data.append(data)
            all_processed_data.append(processed_data)
        self.all_processed_data = all_processed_data
    
    def __len__(self):
        # return len(self.all_indices)
        return self.data_set_length
    
    def reset(self):
        return
        self.epoch_negative_samples = np.random.choice(self.negative_sample_idx, len(self.positive_sample_idx), replace=False)
        # if self.hp['debug']:
            # print(":: Reset dataset")
            # print(self.epoch_negative_samples)
        
        self.epoch_positive_samples = self.positive_sample_idx
    
    def __getitem__(self, data_idx):
        # if self.split == "train":
        #     if data_idx % 2 == 0:
        #         real_idx = self.epoch_negative_samples[data_idx // 2]
        #     else:
        #         real_idx = self.epoch_positive_samples[data_idx // 2]
        # else:
        real_idx = data_idx
        indices = self.all_indices[real_idx]  
        angle_idx = 0 if self.split != "train" else np.random.choice(len(indices))
        idx = indices[angle_idx]
        entry = self.df.iloc[idx]
        
        data = self.all_processed_data[real_idx][angle_idx]
        color_img = np.copy(data['color_img'])
        depth_img = data['depth_img']
        img_mask = data['img_mask']
        mask = data['mask']

        img_height, img_width = self.img_height, self.img_width
        original_height_diff = data['original_height_diff']
        angle_radians = entry['angle'] * (np.pi/4)
        action_angle = F.one_hot(torch.tensor(entry['angle']).long(), self.num_angles)
        action_z_level = F.one_hot(torch.tensor(entry['z_level']).long(), 3)
        
        action_start_pixel = torch.FloatTensor(np.array([data['pix_x'], data['pix_y']]))
        action_end_pixel = torch.FloatTensor(np.array([data['pix_x_dx'], data['pix_y_dy']]))
        if self.hp['distance_to_boundary']:
            x_end, y_end = calculate_action_end(angle_radians, float(entry['length']), self.hp['distance_to_boundary'])
            action_end = torch.FloatTensor(np.array([x_end, y_end]))
        else:
            action_end = action_end_pixel
        
        coins = np.random.rand(3)
        # print(coins)
        if self.split == "train":
            if not self.hp['no_flipping'] and coins[0] >= 0.5:
                color_img = cv2.flip(color_img, 1)
                depth_img = cv2.flip(depth_img, 1)
                img_mask = cv2.flip(img_mask.astype(int), 1) #.astype(bool)
                original_height_diff = cv2.flip(original_height_diff, 1)
                if 'far_depth_start' in data:
                    far_mask = cv2.flip(far_mask, 1)
                if not self.no_gt:
                    mask = cv2.flip(mask, 1)
                new_end_pixel = np.array(unet_utils.flip_end_pixel([data['pix_x'] * img_width, data['pix_y'] * img_width], [data['pix_x_dx'] * img_width, data['pix_y_dy'] * img_width], img_width)) / img_width
                action_end_pixel = torch.FloatTensor(new_end_pixel)
                action_angle_idx = int(((180 - entry['angle'] * 45.0) % 360) / 45)
                action_angle_radians = action_angle_idx * (np.pi/4)
                action_angle = F.one_hot(torch.tensor(action_angle_idx).long(), self.num_angles)
                if self.hp['distance_to_boundary']:
                    x_end, y_end = calculate_action_end(action_angle_radians, float(entry['length']), self.hp['distance_to_boundary'])
                    action_end = torch.FloatTensor(np.array([x_end, y_end]))
                else:
                    action_end = action_end_pixel
            if not self.hp['no_color_jitter'] and coins[1] >= 0.5:
                color_img = Image.fromarray(color_img.astype('uint8'), 'RGB')
                color_img = self.color_jitter_aug(color_img)

        input_img = np.asarray(color_img)
        if input_img.ndim == 2:
            input_img = input_img[np.newaxis, ...]
        else:
            input_img = input_img.transpose((2, 0, 1))
        
        input_img = input_img / 255

        if self.use_height_feature:
            input_img = np.concatenate([input_img, depth_img[np.newaxis, ...]], axis=0)

        
        return_dict = {
            'depth_img' : torch.FloatTensor(depth_img).contiguous(),
            'dataset_idx' : torch.tensor(idx).long(),
            'image': torch.FloatTensor(input_img).contiguous(),
            'image_mask' : torch.as_tensor(img_mask),
            'action_start_pixel' : action_start_pixel,
            'action_end_pixel' : action_end_pixel,
            'action' : torch.cat([action_angle, action_z_level, action_end]),
        }
        
        if not self.no_gt:
            return_dict['mask'] = torch.FloatTensor(mask).contiguous()
        
        if self.use_regression_loss:
            return_dict['height_diff'] = torch.FloatTensor(original_height_diff).contiguous()
        
        return return_dict

class PlantDataset(Dataset):
    def __init__(self, df, split, hp, num_angles=7, no_gt=False):
        self.action_as_image = hp['action_as_image']
        self.split = split
        self.df = df
        self.height_diff_threshold = hp['classes']
        self.num_angles = 12 if hp['rotation_augmentation'] else 7
        self.no_gt = no_gt

        self.hp = hp
        self.use_coordinate_feature = hp['use_coordinate_feature']
        self.use_regression_loss = hp['use_regression_loss']
        self.use_height_feature = hp['use_height_feature']
        self.height_diff_min, self.height_diff_max = hp['height_diff_clip_range']

        color_jitter_b, color_jitter_c, color_jitter_s, color_jitter_h = hp['color_jitter_setting']
        self.color_jitter_aug = torchvision.transforms.ColorJitter(brightness=color_jitter_b, contrast=color_jitter_c, saturation=color_jitter_s, hue=color_jitter_h)
        
        self.img_resolution = df.iloc[0]['resolution']
        self.img_height = self.img_width = int((hp['distance_to_boundary_in_input_image'] / self.img_resolution) * 2)

        if 'data_key' in self.df:
            data_keys = self.df['data_key'].unique()
            all_indices = []
            
            for key in data_keys:
                
                if hp['rotation_augmentation']:
                    indices = df.index[df['data_key'] == key].tolist()
                    all_indices.append(indices)
                else:
                    indices = df.index[(df['data_key'] == key) & (df['angle_idx'] == 0)].tolist()
                    # if len(indices) <= 0:
                    #     indices = sorted(df.index[df['data_key'] == key].tolist())[0]
                    all_indices.append(indices)

        else:
            all_indices = [[idx] for idx in range(len(df))]
        self.all_indices = all_indices

        self.pre_load()

    def pre_load(self):
        all_processed_data = []
        for data_idx in tqdm(range(len(self.all_indices))):
            processed_data = []
            for idx in self.all_indices[data_idx]:
                entry = self.df.iloc[idx]
                folder = entry['folder']
                if 'initial_height_map_path' in entry:
                    initial_height_map_path = entry['initial_height_map_path']  
                    action_end_height_map_path = entry['action_end_height_map_path']
                else:
                    start_frame = entry['start_frame']
                    initial_height_map_path = os.path.join(folder, "height_map", f'{start_frame:05}.pkl')
                    end_frame = entry['frame_num']
                    action_end_height_map_path = os.path.join(folder, "height_map", f'{end_frame:05}.pkl')
                depth_img, color_img, mask_start = pickle.load(open(initial_height_map_path, "rb"))
                img_h, img_w = depth_img.shape
                
                if self.no_gt:
                    img_mask = mask_start.astype(int) # np.zeros((img_h, img_w)).astype(int)
                else: 
                    if action_end_height_map_path:
                        depth_img_end, color_img_end, mask_end = pickle.load(open(action_end_height_map_path, "rb"))
                    
                        img_mask = np.logical_or(mask_start, mask_end).astype(int)
                    
                        height_map_diff_path = entry['height_map_diff_path']
                        original_height_diff = np.load(height_map_diff_path)
 
                    else:
                        depth_img_end, color_img_end, mask_end = np.ones(depth_img) * -1, np.zeros_like(color_img), np.zeros_like(mask_start)
                        img_mask = mask_start.astype(int)
                    
                    if self.hp['close_to_board']:
                        mask = plant_utils.board_revealed_mask(depth_img, color_img, mask_start, depth_img_end, color_img_end, mask_end).astype(int)
                    else:
                        mask = generate_mask(original_height_diff, self.height_diff_threshold)
                        mask[np.where(img_mask)] = 0
                        if not self.hp['no_background']:
                            board_mask= np.load(entry['board_mask_path']).astype(int)
                            mask[np.where(board_mask)] = len(self.height_diff_threshold)
                
                    # lowered = np.copy(mask).astype(bool)
                    # lowered_and_not_valid = np.logical_and(data_end[2], lowered)
                    # mask[np.where(lowered_and_not_valid)] = 0

                depth_img = normalize_height_map(depth_img, min_value=self.hp['height_map_min_value'], value_range=self.hp['height_map_value_range'])
                depth_img[np.where(mask_start)] = 0
                original_height_diff = np.clip(original_height_diff, self.height_diff_min, self.height_diff_max)
                original_height_diff = original_height_diff * 100 
                data = {
                    "depth_img" : depth_img, 
                    "color_img" : color_img, 
                    "img_mask" : img_mask, 
                    "mask" : mask, 
                    "original_height_diff" : original_height_diff,
                }
                if self.hp['use_postaction_mask_loss']:
                    data['postaction_mask'] = mask_end.astype(int)
                if 'far_depth_path_start' in entry:
                    far_depth_start = np.load(entry['far_depth_path_start']).astype(int)
                    data['far_depth_start'] = far_depth_start
                processed_data.append(data)
            all_processed_data.append(processed_data)
        self.all_processed_data = all_processed_data
    
    def __len__(self):
        return len(self.all_indices)
    
    def __getitem__(self, data_idx):
        indices = self.all_indices[data_idx]  
        angle_idx = 0 if self.split != "train" else np.random.choice(len(indices))
        idx = indices[angle_idx]
        entry = self.df.iloc[idx]
        
        data = self.all_processed_data[data_idx][angle_idx]
        color_img = np.copy(data['color_img'])
        depth_img = data['depth_img']
        img_mask = data['img_mask']
        mask = data['mask']
        if 'far_depth_start' in data:
            far_mask = data['far_depth_start']

        img_height, img_width = self.img_height, self.img_width
        original_height_diff = data['original_height_diff']
        angle_radians = entry['angle'] * (np.pi/6)
        if self.hp['action_input_dim'] >= 9:
            action_angle = F.one_hot(torch.tensor(entry['angle']).long(), self.num_angles)
        else:
            action_angle = torch.FloatTensor([entry['angle']])
        
        action_start_pixel = torch.FloatTensor(np.array([entry['pix_x']/img_width, entry['pix_y']/img_height]))
        action_end_pixel = torch.FloatTensor(np.array([entry['pix_x_dx']/img_width, entry['pix_y_dy']/img_height]))
        if self.hp['distance_to_boundary']:
            x_end, y_end = calculate_action_end(angle_radians, float(entry['length']), self.hp['distance_to_boundary'])
            action_end = torch.FloatTensor(np.array([x_end, y_end]))
        else:
            action_end = action_end_pixel
        
        coins = np.random.rand(3)
        # print(coins)
        if self.split == "train":
            if not self.hp['no_flipping'] and coins[0] >= 0.5:
                color_img = cv2.flip(color_img, 1)
                depth_img = cv2.flip(depth_img, 1)
                img_mask = cv2.flip(img_mask.astype(int), 1) #.astype(bool)
                original_height_diff = cv2.flip(original_height_diff, 1)
                if 'far_depth_start' in data:
                    far_mask = cv2.flip(far_mask, 1)
                if not self.no_gt:
                    mask = cv2.flip(mask, 1)
                new_end_pixel = np.array(unet_utils.flip_end_pixel([entry['pix_x'], entry['pix_y']], [entry['pix_x_dx'], entry['pix_y_dy']], img_width)) / img_width
                action_end_pixel = torch.FloatTensor(new_end_pixel)
                action_angle_idx = int(((180 - entry['angle'] * 30.0) % 360) / 30)
                action_angle_radians = action_angle_idx * (np.pi/6)
                if self.hp['action_input_dim'] >= 9:
                    action_angle = F.one_hot(torch.tensor(action_angle_idx).long(), self.num_angles)
                else:
                    action_angle = torch.FloatTensor([action_angle_idx])
                if self.hp['distance_to_boundary']:
                    x_end, y_end = calculate_action_end(action_angle_radians, float(entry['length']), self.hp['distance_to_boundary'])
                    action_end = torch.FloatTensor(np.array([x_end, y_end]))
                else:
                    action_end = action_end_pixel
            if not self.hp['no_color_jitter'] and coins[1] >= 0.5:
                color_img = Image.fromarray(color_img.astype('uint8'), 'RGB')
                color_img = self.color_jitter_aug(color_img)
            # else:
            #     if coins[2] >= 0.5:
            #         color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2Lab)
            if self.hp['remove_far_background'] and coins[2] >= 0.5:
                color_img = np.copy(np.asarray(color_img))
                color_img[:,:,0][np.where(far_mask)] = 0
                color_img[:,:,1][np.where(far_mask)] = 0
                color_img[:,:,2][np.where(far_mask)] = 0
        # if self.hp['remove_far_background']:
        #     color_img[:,:,0][np.where(far_mask)] = 0
        #     color_img[:,:,1][np.where(far_mask)] = 0
        #     color_img[:,:,2][np.where(far_mask)] = 0
        if self.hp['remove_far_background_all']:
            color_img = np.copy(np.asarray(color_img))
            color_img[:,:,0][np.where(far_mask)] = 0
            color_img[:,:,1][np.where(far_mask)] = 0
            color_img[:,:,2][np.where(far_mask)] = 0
        input_img = np.asarray(color_img)
        if input_img.ndim == 2:
            input_img = input_img[np.newaxis, ...]
        else:
            input_img = input_img.transpose((2, 0, 1))
        
        input_img = input_img / 255

        if self.use_height_feature:
            input_img = np.concatenate([input_img, depth_img[np.newaxis, ...]], axis=0)
        
        # print(action_start_x, action_start_y, action_end_x, action_end_y)
        # creating action image
        if self.action_as_image:
            raise "not complete"
        
        if self.use_coordinate_feature:
            raise "not complete"
        
        return_dict = {
            'depth_img' : torch.FloatTensor(depth_img).contiguous(),
            'dataset_idx' : torch.tensor(idx).long(),
            'image': torch.FloatTensor(input_img).contiguous(),
            'image_mask' : torch.as_tensor(img_mask),
            'action_start_pixel' : action_start_pixel,
            'action_end_pixel' : action_end_pixel,
            'action' : torch.cat([action_angle, action_end]),
        }
        if self.hp['use_postaction_mask_loss']:
            mask2 = data['postaction_mask']
            if coins[0] >= 0.5:
                mask2 = cv2.flip(mask2, 1)
            return_dict['postaction_mask'] = torch.FloatTensor(mask2).contiguous()
        if self.hp['predict_precondition']:
            large_force = np.sum(entry['large_force'])
            return_dict['precond_gt'] = torch.tensor(large_force).long()
        if not self.no_gt:
            return_dict['mask'] = torch.FloatTensor(mask).contiguous()
        # if 'pt_x_action_space' in entry and 'pt_y_action_space' in entry:
        #     return_dict['ptx_pty'] = torch.FloatTensor([entry['pt_x_action_space'], entry['pt_y_action_space']])
        # if 'reveal_map_x_start' in entry and 'reveal_map_y_start' in entry:
        #     return_dict['reveal_map_bound'] = torch.FloatTensor([entry['reveal_map_x_start'], entry['reveal_map_y_start']])
        if self.use_regression_loss:
            return_dict['height_diff'] = torch.FloatTensor(original_height_diff).contiguous()
        return return_dict