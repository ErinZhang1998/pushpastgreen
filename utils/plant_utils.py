import cv2
import pickle 
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt 
from autolab_core import Point
import collections
import os
import pandas as pd
from autolab_core import YamlConfig, RigidTransform, PointCloud

def collect_all_action_df(root_folders):
    df_list = []
    
    for root_folder,csv_file in root_folders:
        for folder in os.listdir(root_folder):
            if folder.endswith(".zip"):
                continue
            folder_full_path = os.path.join(root_folder, folder)
            csv_file_path = os.path.join(folder_full_path, csv_file)
            if not os.path.exists(csv_file_path):
                continue
            try:
                df = pd.read_csv(csv_file_path)
                df_list.append(df)
                
            except:
                continue
    print(len(df_list))
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all, df_list

def create_split(length, split_ratio):
    np.random.seed(1119)
    indices = np.arange(length)
    np.random.shuffle(indices)

    num_train = int(len(indices) * split_ratio[0])
    num_val = int(len(indices) * split_ratio[1])
    num_test = len(indices) - num_train - num_val
    df_split = ["Null"] * length
    for i in range(length):
        if i < num_train:
            df_split[indices[i]] = "Train"
        elif i >= num_train and i < num_train+num_val:
            df_split[indices[i]] = "Val"
        else:
            df_split[indices[i]] = "Test"
    return df_split

class AxIter:
    def __init__(self, axes):
        self.axes = []
        for a in axes:
            
            if type(a) is np.ndarray:
                for b in a:
                    self.axes.append(b)
            else:
                self.axes.append(a)
        self.current = -1
        self.high = len(self.axes)
                

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            return self.axes[self.current]
        raise StopIteration

def save_depth_as_png(depth, depth_path, scale_factor = 1000):
    depth_scaled = depth * scale_factor
    depth_scaled[depth_scaled > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    depth_img = depth_scaled.astype(np.uint16)
    if depth_path is None:
        return depth_img
    cv2.imwrite(depth_path, depth_img)

def read_depth_image(depth_path):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH ).astype(np.float64)
    depth_image = depth_image / 1000.0
    return depth_image

def average_depth(depth_images):
    # mask_acc = np.zeros_like(depth_images[0])
    # sum_acc = np.zeros_like(depth_images[0])
    # for depth_image in depth_images:
    #     invalid_mask = np.logical_not(np.isclose(depth_image, 0.0, rtol=0.0, atol=1e-4)).astype(int)
    #     mask_acc += invalid_mask
    #     sum_acc += depth_image
    # return sum_acc / (mask_acc + 1e-5)
    mask_acc = np.zeros_like(depth_images[0])
    sum_acc = []
    for depth_image in depth_images:
        invalid_mask = np.logical_not(np.isclose(depth_image, 0.0, rtol=0.0, atol=1e-4)).astype(int)
        mask_acc += invalid_mask
        sum_acc.append(depth_image)
    print(":: #depth images used for calculating median ", len(sum_acc))
    return np.median(np.stack(sum_acc), axis=0)

def get_pcd_from_color_and_depth_image_paths(color_path, depth_path, intrinsic):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def get_pcd_from_rgb_depth_images(rgb,depth,intr):
    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr)
    return pcd

def get_averaged_depth_images(depth_image_folder, path_template, frame_num, num_frames_to_average=5):
    depth_images = []

    fnum = frame_num - num_frames_to_average//2
    tally = num_frames_to_average
    inc = 1
    flag = False
    frames_so_far = []
    loop_times = 0
    while tally > 0:
        fnum_prefix = path_template.format(f'{fnum:05}')
        fnum_path = os.path.join(depth_image_folder, f'{fnum_prefix}.png')
        loop_times += 1
        if loop_times > num_frames_to_average * 2:
            print(":: BUG in calculating median depth ", loop_times)
            break
        if not os.path.exists(fnum_path):
            print(":: Failed to read: ", fnum_path)
            if fnum >= frame_num + num_frames_to_average//2:
                fnum = frames_so_far[0]
                inc = -1
                flag = True 
            # if fnum <= frame_num - num_frames_to_average//2:
            #     if flag:
            #         break
            fnum += inc
            continue 
        print(f":: Averaging with depth {fnum_path}")
        depth_images.append(read_depth_image(fnum_path))
        tally -= 1
        frames_so_far.append(fnum)
        fnum += inc
        
    if len(depth_images) % 2 == 0:
        depth_images = depth_images[:-1]
    depth_full = average_depth(depth_images)
    return depth_full

def get_averaged_depth_images_from_stack_of_depth_images(stack_of_depth_images, frame_num, num_frames_to_average=5):
    depth_images = []
    fnum = frame_num - num_frames_to_average//2
    tally = num_frames_to_average
    inc = 1
    flag = False
    frames_so_far = []
    loop_times = 0
    while tally > 0:
        loop_times += 1
        if loop_times > num_frames_to_average * 2:
            print(":: BUG in calculating median depth ", loop_times)
            break
        if fnum < 0 or fnum >= len(stack_of_depth_images):
            print(f":: Failed to read No. {fnum}")
            if fnum >= frame_num + num_frames_to_average//2:
                fnum = frames_so_far[0]
                inc = -1
                flag = True 
            # if fnum <= frame_num - num_frames_to_average//2:
            #     if flag:
            #         break
            fnum += inc
            continue 
        print(f":: Averaging with depth No. {fnum}")
        depth_images.append(stack_of_depth_images[fnum].astype(np.float64) / 1000.0)
        tally -= 1
        frames_so_far.append(fnum)
        fnum += inc
        
    if len(depth_images) % 2 == 0:
        depth_images = depth_images[:-1]
    
    depth_full = average_depth(depth_images)
    depth_full = save_depth_as_png(depth_full, None, scale_factor = 1000)
    return depth_full

def generate_depth_and_height_image_control_background(pcd, intrinsic_iam, transform_to_height_image_frame, ground_height, plant_height_max,  plant_depth_max, too_far_height, pcd_rs_world = None):
        
    pcd_points = np.asarray(pcd.points)
    sort_z_ind = np.argsort(pcd_points[:,2])[::-1]
    pcd_points = pcd_points[sort_z_ind]
    pcd_pts = PointCloud(pcd_points.T, frame='azure_kinect_overhead')
    x_coords, y_coords = intrinsic_iam.project(pcd_pts).data
    depth_image_from_intr = intrinsic_iam.project_to_image(pcd_pts).data

    # if a point cloud is transformed, then the coordinates might be outside of the image boundary
    x_in_range = np.logical_and(x_coords >= 0, x_coords < intrinsic_iam.width)
    y_in_range = np.logical_and(y_coords >= 0, y_coords < intrinsic_iam.height)
    in_range = np.logical_and(x_in_range, y_in_range)
    x_coords = x_coords[in_range]
    y_coords = y_coords[in_range]

    # transform the pcd to be in upright frame
    pcd2 = transform_to_height_image_frame(pcd)
    pcd_points_in_frontal_parallel_frame = np.asarray(pcd2.points)[sort_z_ind][in_range]
    pcd_points_in_frontal_parallel_frame[:,2] -= ground_height
    
    # this to take out points in the robot arm, might still have points on the grabber
    if pcd_rs_world:
        height_in_range = pcd_points_in_frontal_parallel_frame[:,2] < plant_height_max
    else:
        height_in_range = np.ones(len(pcd_points_in_frontal_parallel_frame)).astype(bool)
    x_coords = x_coords[height_in_range]
    y_coords = y_coords[height_in_range]
    
    pcd_points = pcd_points[in_range][height_in_range]
    depth_image = np.zeros((intrinsic_iam.height,intrinsic_iam.width))
    depth_image[y_coords, x_coords] = pcd_points[:,2]
    depth_image_orig = np.copy(depth_image)
    # to obtain the points correspond to the ground and the table stand
    too_far = depth_image > plant_depth_max

    pcd3 = transform_to_height_image_frame(pcd)
    pcd_points_in_frontal_parallel_frame = np.asarray(pcd3.points)[sort_z_ind][in_range][height_in_range]
    pcd_points_in_frontal_parallel_frame[:,2] -= ground_height
    new_depth_image = np.zeros((intrinsic_iam.height,intrinsic_iam.width))
    new_depth_image[y_coords, x_coords] = pcd_points_in_frontal_parallel_frame[:,2]
    # import pdb;pdb.set_trace()
    new_depth_image[np.where(too_far)] = too_far_height

    # if debug:
    #     fig,ax = plt.subplots(1,2, figsize=(20,10))
    #     ax[0].imshow(depth_image_orig)
    #     ax[1].imshow(new_depth_image)
    #     plt.show()
    
    return depth_image_orig, new_depth_image

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def from_3d_array_to_2d_coord_array(coords, workspace_limits_array, image_shape, resolution, restrict_to_within_image=True):
    """
    :param coords: (B, 2)
    :param workspace_limits_array: (B, 2, 2) or (B, 3, 2)
    :param image_shape: [height, width]
    """
    pix_xs = np.floor((coords[:,0] - workspace_limits_array[:, 0, 0])/resolution).astype(int) # (B, )
    pix_ys = np.floor((coords[:,1] - workspace_limits_array[:, 1, 0])/resolution).astype(int) # (B, )
    pix_ys = image_shape[0] - 1 - pix_ys
    if restrict_to_within_image:
        pix_xs[pix_xs < 0] = 0
        pix_ys[pix_ys < 0] = 0
        pix_xs[pix_xs >= image_shape[1]] = image_shape[1]-1
        pix_ys[pix_ys >= image_shape[0]] = image_shape[0]-1
    return pix_xs, pix_ys

def get_coordinate_in_image(coords, xmin, ymin, width, height, resolution, restrict_to_within_image=True, pixel_include_upper=True):
    """
    :param coords: (N, >=2) first column is x coordinate, second column is y coordinate
    :param xmin: x value of lower left corner of an image  
    :param ymin: 
    """
    pix_xs = np.floor((coords[:,0] - xmin)/resolution).astype(int)
    pix_ys = np.floor((coords[:,1] - ymin)/resolution).astype(int)
    pix_ys = height - 1 - pix_ys
    if not pixel_include_upper:
        pix_ys += 1
    if restrict_to_within_image:
        pix_xs[pix_xs < 0] = 0
        pix_ys[pix_ys < 0] = 0
        pix_xs[pix_xs >= width] = width-1
        pix_ys[pix_ys >= height] = height-1
    return pix_xs, pix_ys

def from_workspace_limits_to_2d_dimension(workspace_limits, heightmap_resolution):
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)
    return heightmap_size

def get_heightmap(color_pts, surface_pts, workspace_limits, heightmap_resolution, z_min_max = True, median_blur = False):
    
    # Compute heightmap size
    heightmap_size = from_workspace_limits_to_2d_dimension(workspace_limits, heightmap_resolution)
    # print(heightmap_size)
    if z_min_max:
        sort_z_ind = np.argsort(surface_pts[:,2])
    else:
        sort_z_ind = np.argsort(surface_pts[:,2])[::-1]
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    # 
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind_xy = np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1])
    if z_min_max:
        heightmap_valid_ind = np.logical_and(heightmap_valid_ind_xy, surface_pts[:,2] < workspace_limits[2][1])
    else:
        heightmap_valid_ind = np.logical_and(heightmap_valid_ind_xy, surface_pts[:,2] > workspace_limits[2][0])
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    
    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    
    # heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    # heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    # heightmap_pix_x[heightmap_pix_x >= heightmap_size[1]] = heightmap_size[1]-1
    # heightmap_pix_y[heightmap_pix_y >= heightmap_size[0]] = heightmap_size[0]-1
    # heightmap_pix_y = heightmap_size[0] - 1 - heightmap_pix_y
    heightmap_pix_x, heightmap_pix_y = get_coordinate_in_image(surface_pts, workspace_limits[0][0], workspace_limits[1][0], heightmap_size[1], heightmap_size[0], heightmap_resolution, restrict_to_within_image=True)
    
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    
    if median_blur:
        depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
        # plt.imshow(depth_heightmap)
        # plt.show()
        depth_scaled = depth_heightmap * -100
        print("WARNING, might overflow for values that are negative and over 255: ", np.sum(depth_scaled > np.iinfo(np.uint8).max), depth_heightmap.shape[0] * depth_heightmap.shape[1])
        depth_scaled[depth_scaled > np.iinfo(np.uint8).max] = np.iinfo(np.uint8).max
        depth_img = depth_scaled.astype(np.uint8)
        
        depth_heightmap = cv2.medianBlur(depth_img,7).astype(np.float64) / -100.0
        # depth_heightmap = cv2.medianBlur(depth_heightmap.astype(np.float32),5)
        # plt.imshow(depth_heightmap)
        # plt.show()
        
        # import pdb;pdb.set_trace()
    if z_min_max:
        z_bottom = workspace_limits[2][0]
        depth_heightmap = depth_heightmap - z_bottom
    else:
        z_bottom = workspace_limits[2][1]
        depth_heightmap = z_bottom - depth_heightmap
    mask1 = np.isclose(depth_heightmap, -workspace_limits[2][0], rtol=0.0, atol=1e-03) # z value is close to 0, -1mm to 1mm
    mask2 = depth_heightmap < 0 # z value is lower than the minimum
    # import pdb;pdb.set_trace()
    height_map_mask = np.logical_or(mask1, mask2)
    depth_heightmap[height_map_mask] = 0 # depth = 0 could be no point at this location
    return color_heightmap, depth_heightmap, height_map_mask

def find_min_max_frame(folder):
    process_frames = []
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            process_frames.append(int(file.split(".")[0]))
    process_frames = sorted(process_frames)
    initial_height_map_path = os.path.join(folder, f'{process_frames[0]:05}.pkl')
    action_end_height_map_path = os.path.join(folder, f'{process_frames[-1]:05}.pkl')
    return process_frames[0], process_frames[-1]

def get_all_actions(folder):
    actions = {}
    for file in os.listdir(folder):
        if not file.endswith("robot_action.pkl") and (not file.endswith("robot.pkl")) :
            continue
        try:
            action = pickle.load(open(os.path.join(folder, file), "rb"))
            start_frame = int(action['start_frame'])
        except:
            continue
        last_frame = int(action['action_frame_infos'][-1])
        if "t" in action and 'i_traj' in action:
            i_traj = action['i_traj']
            t = action['t']
            path_template = f'{i_traj:05}_{t:05}' + '_{}'
            k = f'{i_traj:05}_{t:05}_{start_frame:05}_{last_frame:05}'
        else:
            path_template = '{}'
            k = f'{start_frame:05}_{last_frame:05}'
        action['path_template'] = path_template
        actions[k] = action
    if len(actions) == 0:
        print(f"WARNING no actions found in: {folder}, so it will be removed")
        # os.rmdir(folder)
    return actions

## HARD_CODED
def get_nov_11_workspace():
    # workspace_limits = np.asarray(config['workspace_limits_big']).copy()
    # workspace_limits = np.asarray([[-0.30362051271863066, 0.6963794872813693], [-0.7476710762011101, 0.002328923798889937], [-0.6418703088587832, -0.4841268978523616]])
    # workspace_limits[0][0] += 0.13
    # workspace_limits[0][1] = workspace_limits[0][0] + 150 * 0.005
    # workspace_limits[1][1] = workspace_limits[1][0] + 150 * 0.005
    workspace_limits = np.asarray([[-0.17362051,  0.57637949],
        [-0.74767108,  0.00232892],
        [-0.64187031, -0.4841269 ]])
    return workspace_limits,150,150

nov_11_angle_to_angle_idx_list = {0: 0, 30:1, 60:2, 90:3,120:4,150:5,180:6}
folder_name_map = {
    "/scratch0/zhang401local" : ["/data11/zhang401", "/data01/zhang401", "/data02/zhang401"],
    "/home/zhang401local" : ["/home/zhang401"],
}

def convert_path(path):
    if os.path.exists(path):
        return path
    else:
        for k,v in folder_name_map.items():
            if k in path:
                tried = ""
                for possible_path in v:
                    new_path = path.replace(k,possible_path)
                    if os.path.exists(new_path):
                        return new_path     
                    else:
                        tried += f" {new_path}"           
                raise ValueError(f'Tried to replace {k} with {tried}, but none exists.')

def get_actions_parameter_range(pt_xs, pt_ys, angles, workspace_limits, in_radians = False):
    """
    :param pt_xs: array 
    :param pt_ys: array 
    :param angles: array 
    """
    max_lengths = np.zeros_like(pt_xs)
    sample_xmins = workspace_limits[0][0] - pt_xs
    sample_xmaxs = workspace_limits[0][1] - pt_xs
    sample_ymins = workspace_limits[1][0] - pt_ys
    sample_ymaxs = workspace_limits[1][1] - pt_ys

    if not in_radians:
        angles_rad = np.radians(angles)
    else:
        angles_rad = angles 

    x_units = np.cos(angles_rad)
    y_units = np.sin(angles_rad)
    
    atol_value = 1e-04
    non_x_zero = ~np.isclose(x_units, 0.0, rtol=0.0, atol=atol_value)
    non_y_zero = ~np.isclose(y_units, 0.0, rtol=0.0, atol=atol_value)
    x_units_dummy = np.copy(x_units)
    x_units_dummy[~non_x_zero] = 1e-05
    y_units_dummy = np.copy(y_units)
    y_units_dummy[~non_y_zero] = 1e-05
    case_1 = sample_xmaxs / x_units_dummy
    case_2 = sample_xmins / x_units_dummy
    case_3 = sample_ymaxs / y_units_dummy
    case_4 = sample_ymins / y_units_dummy
    case_1[case_1 < 0] = 1e5
    case_2[case_2 < 0] = 1e5
    case_3[case_3 < 0] = 1e5
    case_4[case_4 < 0] = 1e5
    max_lengths = np.minimum(np.minimum(np.minimum(case_1, case_2), case_3), case_4)
    
   
    angle_0_criteria = np.isclose(y_units, 0.0, rtol=0.0, atol=atol_value) & (x_units > 0)
    angle_180_criteria = np.isclose(y_units, 0.0, rtol=0.0, atol=atol_value) & (x_units < 0)
    angle_90_criteria = np.isclose(x_units, 0.0, rtol=0.0, atol=atol_value) & (y_units > 0)
    angle_270_criteria = np.isclose(x_units, 0.0, rtol=0.0, atol=atol_value) & (y_units < 0)
    max_lengths[angle_0_criteria] = np.abs(sample_xmaxs[angle_0_criteria])
    max_lengths[angle_180_criteria] = np.abs(sample_xmins[angle_180_criteria])
    max_lengths[angle_90_criteria] = np.abs(sample_ymaxs[angle_90_criteria])
    max_lengths[angle_270_criteria] = np.abs(sample_ymins[angle_270_criteria])

    return max_lengths

def get_action_parameter_range(pt_x, pt_y, workspace_limits, angle_list = None, in_radians = False):
    sample_xmin = workspace_limits[0][0] - pt_x
    sample_xmax = workspace_limits[0][1] - pt_x
    sample_ymin = workspace_limits[1][0] - pt_y
    sample_ymax = workspace_limits[1][1] - pt_y
    if angle_list is None:
        angle_list = [0, 30, 60, 90, 120, 150, 180]
    # np.arange(num_angles+1) * (angle_max / num_angles)
    max_length_list = []
    end_xy_points = []
    
    angle_list_final = []
    for angle in angle_list:
        # if np.isclose(angle, 90, rtol=0.0, atol=1e-04):
        #     continue
        
        angle_list_final.append(angle)
        if not in_radians:
            angle_rad = np.radians(angle)
        else:
            angle_rad = angle
        x_unit = np.cos(angle_rad)
        y_unit = np.sin(angle_rad)
        
        if np.isclose(y_unit, 0.0, rtol=0.0, atol=1e-04):
            if x_unit < 0:
                max_length = np.abs(sample_xmin)
            else:
                max_length = np.abs(sample_xmax)
        elif np.isclose(x_unit, 0.0, rtol=0.0, atol=1e-04):
            if y_unit < 0:
                max_length = np.abs(sample_ymin)
            else:
                max_length = np.abs(sample_ymax)

        elif x_unit < 0:
            if y_unit < 0:
                max_length = min(sample_ymin / y_unit, sample_xmin / x_unit)
            else:
                max_length = min(sample_ymax / y_unit, sample_xmin / x_unit)
        elif x_unit > 0:
            if y_unit < 0:
                max_length = min(sample_ymin / y_unit, sample_xmax / x_unit)
            else:
                max_length = min(sample_ymax / y_unit, sample_xmax / x_unit)
        
        end_xy_points.append([max_length * x_unit, max_length * y_unit])
        max_length_list.append(max_length)
    
    return np.asarray(angle_list_final), max_length_list, end_xy_points

def from_rgb_depth_path_to_height_map(rgb_path, depth_path, intrinsics, pcd_post_process, workspace_limits, resolution):
    raise
    pcd = get_pcd_from_color_and_depth_image_paths(rgb_path, depth_path, intrinsics)
    pcd = pcd_post_process(pcd)
    color_pts = np.round(np.array(pcd.colors) * 255).astype(np.uint8)
    surface_pts = np.array(pcd.points)
    color_heightmap, height_map, height_map_mask = get_heightmap(color_pts, surface_pts, workspace_limits, resolution)
    
    return color_heightmap, height_map, height_map_mask

def from_rgb_depth_image_to_color_surface_points(rgb_image, depth_image, pcd_post_process, intrinsics):
    color_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    pcd = pcd_post_process(pcd)
    color_pts = np.round(np.array(pcd.colors) * 255).astype(np.uint8)
    surface_pts = np.array(pcd.points)
    return color_pts, surface_pts

def from_space_limits_to_four_corners_at_zmin(space_limits):
    '''
    :param space_limits: (2, 3) [xmin,ymin,zmin],[xmax,ymax,zmax]
    :return four_corners: (4, 3) 
    '''
    xmin,ymin,zmin = space_limits[:,0]
    xmax,ymax,zmax = space_limits[:,1]

    four_corners = np.asarray([
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
    ])
    return four_corners

def project_3d_to_2d(pt, intrinsics, frame_name):
    '''
    :param pt: list or array of size 3
    :param intrinsics: type CameraIntrinsics
    :param frame_name: name of the camera frame
    :return pt_uv: array of size 2 [x_coord, y_coord]
    '''
    pt = np.asarray(pt)
    pt_3d = Point(pt, frame=frame_name)
    pt_uv = intrinsics.project(pt_3d).data
    return pt_uv

def homography_transform_color_and_depth_image(color_image, depth_image, M, new_image_w, new_image_h):
    
    color_image_transformed = cv2.warpPerspective(np.asarray(color_image), \
                                                  M, (new_image_w, new_image_h), flags=cv2.INTER_LINEAR)

    depth_image_transformed = cv2.warpPerspective(np.asarray(depth_image), \
                                                  M, (new_image_w, new_image_h), flags=cv2.INTER_LINEAR)
    return color_image_transformed, depth_image_transformed

def get_global_workspace_limits(workspace_limits, distance_to_boundary, workspace_limits_big =None, base_height=None):
    # hardcoded_distance = distance_to_boundary - 0.01
    diff_from_workspace_limits = np.abs(workspace_limits_big - workspace_limits) - 0.25
    diff_from_workspace_limits[diff_from_workspace_limits > 0] = 0.0
    diff_from_workspace_limits[:,1] *= -1.0
    diff_from_workspace_limits[2] = 0.0
    workspace_limits_expanded = np.copy(workspace_limits_big)
    workspace_limits_expanded = workspace_limits_expanded + diff_from_workspace_limits
    if base_height:
        workspace_limits_expanded[2][0] = base_height
    else:
        workspace_limits_expanded[2][0] = workspace_limits[2][0] + 0.12
    
    # workspace_limits_expanded = np.copy(workspace_limits)
    # workspace_limits_expanded[:,0] -= hardcoded_distance
    # workspace_limits_expanded[:,1] += hardcoded_distance
    # if base_height:
    #     workspace_limits_expanded[2][0] = base_height
    # else:
    #     workspace_limits_expanded[2][0] = workspace_limits[2][0] + 0.12
    return workspace_limits_expanded

def get_board_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # green_mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    brown_mask = cv2.inRange(hsv, (89, 65, 90),(116, 255, 255))
    # 206, 183, 151
    ## final mask and masked
    # mask = cv2.bitwise_or(mask1, mask2)
    # target = cv2.bitwise_and(img, img, mask=brown_mask)
    return (brown_mask / 255).astype(int)

def board_showing(start_height_map, start_color_heightmap, start_height_map_mask):
    close_to_board = np.isclose(start_height_map, 0, rtol=0.0, atol=0.005)
    
    board = get_board_mask(start_color_heightmap)
    
    all = board#np.logical_or(close_to_board, board)
    all[np.where(start_height_map_mask)] = False
    return all

def board_revealed_mask(start_height_map, start_color_heightmap, start_height_map_mask, \
                       end_height_map, end_color_heightmap, end_height_map_mask):
    
    all_masks = np.logical_or(start_height_map_mask, end_height_map_mask)
    
    h1_close_to_board = np.isclose(start_height_map, 0, rtol=0.0, atol=0.005)
    h2_close_to_board = np.isclose(end_height_map, 0, rtol=0.0, atol=0.005)
    h2_close_to_board[np.where(h1_close_to_board)] = False
    
    h1_board = get_board_mask(start_color_heightmap)
    h2_board = get_board_mask(end_color_heightmap)
    h2_board[np.where(h1_board)] = False
    
    h2_all = h2_board#np.logical_or(h2_close_to_board, h2_board)
    h2_all[np.where(all_masks)] = False
    return h2_all

def get_action_start_frame(action):
    return int(action['start_frame'])

def get_action_end_frame(action):
    return int(action['action_frame_infos'][-3]) if len(action['action_frame_infos']) == 5 else int(action['action_frame_infos'][-1]-1)

def get_frames_to_process(action, only_start_end):
    start_frame = int(action['start_frame'])
    last_frame = int(action['action_frame_infos'][-1])
    if only_start_end:
        frames_to_process = [start_frame] + [int(f_n) for f_n in action['action_frame_infos'][:-1]] + [last_frame-1]
        # frames_to_process += list(range(action['action_frame_infos'][-3]+1, action['action_frame_infos'][-3]+11))
        # frames_to_process[-3] += 9
    else:
        frames_to_process = list(range(start_frame, last_frame))
    return frames_to_process

def get_global_height_map_suffix(img_input_resolution):
    img_input_resolution_m = int(img_input_resolution * 1000 )
    return f"{img_input_resolution_m}mm"

def get_all_start_pt_in_saved_data(actions, T_world_to_sampled_action_space=None, grabber_z_offset = 0.293):
    
    processed_actions = []
    order = []
    for k,action in actions.items():
        if T_world_to_sampled_action_space and ('start_pt_world_real' in action):
            ptt = np.copy(np.asarray(action['start_pt_world_real']))
            ptt[-1] -= grabber_z_offset # (0.4-0.107)
            pt_x,pt_y,pt_z = (T_world_to_sampled_action_space * Point(ptt, frame="world")).data
            
        else:
            pt_x,pt_y,pt_z = action["start_pt"]
        # length = np.linalg.norm(np.asarray(action['end_pt']) - np.asarray(action["start_pt"]))
        processed_actions.append([pt_x,pt_y,pt_z,action['angle'],action['length_real']])
        order.append(k)
    return np.stack(processed_actions), order

def get_rotate_image_matrix(width, height, angle):
    corners = np.array([[0,0,1],[0,height-1,1],[width-1,0,1],[width-1,height-1,1]])
    rotate_matrix = cv2.getRotationMatrix2D((width/2,height/2), angle=angle, scale=1)
    rotate_matrix = np.concatenate((rotate_matrix, np.array([[0,0,1]])))
    corners_rotated = (rotate_matrix @ corners.T).T
    xmin,ymin,_ = np.min(corners_rotated, 0)
    xmax,ymax,_ = np.max(corners_rotated, 0)
    new_width = np.ceil(xmax - xmin)
    new_height = np.ceil(ymax - ymin)
    translate_matrix = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
    matrix = translate_matrix @ rotate_matrix
    corners_rotated_translated = (matrix @ corners.T).T
    xmin2,ymin2,_ = np.min(corners_rotated_translated,0)
    return matrix, new_width, new_height, xmin2, ymin2

def down_sample_point_cloud(pcd, voxel_size, remove_statistical_outlier_nb_neighbor, remove_statistical_outlier_std_ratio):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    _, inc_ind = pcd_down.remove_statistical_outlier(nb_neighbors=remove_statistical_outlier_nb_neighbor, std_ratio=remove_statistical_outlier_std_ratio)
    pcd_down = pcd_down.select_by_index(inc_ind)
    return pcd_down

def preprocess_point_cloud(pcd, config):
    # pcd_precompute = o3d.io.read_point_cloud(config['pcd_crop_path'])
    # pcd_precompute_obb = pcd_precompute.get_axis_aligned_bounding_box()
    # pcd_precompute_obb.color = (1,0,0)
    voxel_size = config['voxel_size']
    pcd_down = down_sample_point_cloud(pcd, voxel_size, config['remove_statistical_outlier_nb_neighbor'], config['remove_statistical_outlier_std_ratio'])
    
    # pcd_down = pcd_down.crop(pcd_precompute_obb)
    radius_normal = voxel_size * config['radius_normal_to_voxel_ratio']
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=config['normal_max_nn']))

    radius_feature = voxel_size * config['radius_feature_to_voxel_ratio']
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=config['fpfh_feature_max_nn']))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, global_registration_distance_threshold_voxel_ratio = 1.5):
    distance_threshold = voxel_size * global_registration_distance_threshold_voxel_ratio
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99))
    return result

def median_blur_height_map(depth_heightmap):
    depth_scaled = depth_heightmap * 100
    print("WARNING, might overflow for values that are negative and over 255: ", np.sum(depth_scaled > np.iinfo(np.uint8).max), depth_heightmap.shape[0] * depth_heightmap.shape[1])
    depth_scaled[depth_scaled > np.iinfo(np.uint8).max] = np.iinfo(np.uint8).max
    depth_img = depth_scaled.astype(np.uint8)
    
    blurred_heightmap = cv2.medianBlur(depth_img,5).astype(np.float64) / 100.0

    return blurred_heightmap