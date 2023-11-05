import time
import copy
import random
import numpy as np 
import open3d as o3d
import cv2
import rospy
from frankapy import FrankaArm
from frankapy import FrankaConstants as FC
from autolab_core import YamlConfig, RigidTransform, Point, PointCloud
from franka_interface_msgs.msg import RobotState
import message_filters
from sensor_msgs.msg import Image, CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError

import utils.plant_utils as plant_utils

from perception import CameraIntrinsics

def get_duration(dist):
    
    if dist <= 0.2:
        return 3
    elif 0.2 <= dist < 0.3:
        return 4
    elif 0.3 <= dist < 0.4:
        return 5
    elif 0.4 <= dist < 0.5:
        return 7
    else:
        return 9

class HParams:
    def __init__(self) -> None:
        pass

class RobotBaseEnv:
    def __init__(self, env_params):
        yaml_config = dict(YamlConfig(env_params['yaml_config_path']).config)
        self._hp = self._default_hparams()
        for name, value in yaml_config.items():
            self._hp[name] = value
        for name, value in env_params.items():
            self._hp[name] = value
        
        
        if self._hp['icp_yaml_file'] and self._hp['icp_yaml_file'] != '':
            print(f":: Icp config file from path={self._hp['icp_yaml_file']}")
            icp_yaml_config = dict(YamlConfig(env_params['icp_yaml_file']).config)
            self._hp['icp_config'] = icp_yaml_config
        
        self.cv_bridge = CvBridge()
        self.initial_joints = np.load(self._hp['initial_joints'])
        if not self._hp["no_robot"]:
            self.fa = FrankaArm(with_gripper=False)
            # self.reset_robot()
        
        # intrinsics
        intrinsic_iam = CameraIntrinsics.load(self._hp['intrinsics_file_path'])
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(intrinsic_iam.width, intrinsic_iam.height, intrinsic_iam.fx, intrinsic_iam.fy, intrinsic_iam.cx, intrinsic_iam.cy)
        self.intrinsic_iam = intrinsic_iam
        # bounding box
        pcd_precompute = o3d.io.read_point_cloud(self._hp['plant_space_pcd'])
        
        if self._hp['crop_scene_with_aabb']:
            self.plant_space_obb = pcd_precompute.get_axis_aligned_bounding_box()
        else:
            self.plant_space_obb = pcd_precompute.get_oriented_bounding_box()
            
        self.plant_space_obb.color = (1,0,1)

        pcd_workspace = o3d.io.read_point_cloud(plant_utils.convert_path(self._hp['workspace_pcd']))
        pcd_workspace_aabb = pcd_workspace.get_axis_aligned_bounding_box()
        crop_max_bound = pcd_workspace_aabb.get_max_bound()
        aabb_max_bound = np.asarray([crop_max_bound[0]+0.1, crop_max_bound[1]+0.1, crop_max_bound[2]])
        aabb_min_bound = pcd_workspace_aabb.get_min_bound()-0.1
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound = aabb_min_bound, max_bound = aabb_max_bound)
        aabb.color = (0,0,1)
        self.icp_aabb = aabb

        # transformations
        self.T_flip = RigidTransform(rotation=np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]),translation=np.array([0.0,0.0,0.0]), from_frame='azure_kinect_overhead', to_frame='azure_kinect_overhead_flipped')  
        self.T_camera_cameraupright = RigidTransform.load(self._hp['plant_upright_transformation_path'])
        self.T_camera_cameraupright_height = RigidTransform.load(self._hp['plant_upright_transformation_path_height'])
        self.T_camera_world = RigidTransform.load(self._hp['extrinsics_file_path'])   
        self.T_alignplant = RigidTransform.load(self._hp['plant_ee_align_transformation_path'])
        self.plant_angle_radians = float(self._hp['plant_angle']) if self._hp['plant_angle'] else None
        self.T_sampled_action_space_to_world = self.T_camera_world * self.T_flip.inverse() * self.T_camera_cameraupright.inverse()
        self.T_world_to_sampled_action_space = self.T_camera_cameraupright * self.T_flip * self.T_camera_world.inverse()
        
        self.workspace_limits = np.asarray(self._hp['workspace_limits'])
        self.actionspace_limits = np.asarray(self._hp['workspace_limits'])

        if self._hp['use_rs']:
            intrinsic_iam_rs = CameraIntrinsics.load(plant_utils.convert_path(self._hp['intrinsics_file_rs_path']))
            intrinsic_o3d_rs = o3d.camera.PinholeCameraIntrinsic(intrinsic_iam_rs.width, intrinsic_iam_rs.height, intrinsic_iam_rs.fx, intrinsic_iam_rs.fy, intrinsic_iam_rs.cx, intrinsic_iam_rs.cy)
            self.intrinsic_o3d_rs = intrinsic_o3d_rs
            self.intrinsic_iam_rs = intrinsic_iam_rs

            self.T_rs_ee = RigidTransform.load(plant_utils.convert_path(self._hp['rs_to_ee_path']))

        # def board_to_camera_frame(pt):
        #     pt = Point(np.asarray(pt), frame="azure_kinect_overhead_upright")
        #     pt_new = self.T_flip.inverse() * self.T_camera_cameraupright.inverse() * pt
        #     return pt_new.data
        self.recorded_frames = []
        
        self.prepare_frame_collector()
        
    
    def _callback(self, robot_info, rgb_image, depth_image):
        rgb_cv_image = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(rgb_image, "rgb8")).astype(np.uint8)
        depth_cv_image = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(depth_image)).astype(np.uint16)
        
        self.recorded_frames.append([robot_info, rgb_cv_image, depth_cv_image])
    
    def _callback_with_rs(self, x0,x1,x2,x3,x4):
        rgb_cv_image = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(x1, "rgb8")).astype(np.uint8)
        depth_cv_image = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(x2)).astype(np.uint16)
        rs_rgb = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(x3, "rgb8")).astype(np.uint8)
        rs_depth = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(x4)).astype(np.uint16)
        robot_ee_pose_matrix = np.asarray(x0.O_T_EE).reshape((4,4)).T
        self.recorded_frames.append([x0,rgb_cv_image, depth_cv_image,rs_rgb,rs_depth, robot_ee_pose_matrix])

    
    def reset_robot(self):
        # input("Going to initial joint configuration.")
        self.fa.reset_joints(duration=5)
        self.fa.goto_joints(self.initial_joints, ignore_virtual_walls=True)
    
    def prepare_frame_collector(self):
        robot_sub = message_filters.Subscriber(self._hp['robot_state_topic'], RobotState)
        image_sub = message_filters.Subscriber(self._hp['color_topic'], Image)
        depth_sub = message_filters.Subscriber(self._hp['depth_topic'], Image)
        self.subscribers = [robot_sub,image_sub,depth_sub]
        if self._hp['use_rs']:
            image_rs_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
            depth_rs_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
            print(":: Using images from realsense camera")
            self.subscribers += [image_rs_sub,depth_rs_sub]
            callback_fn = lambda x,y,z,a,b : self._callback_with_rs(x,y,z,a,b)
        else:
            callback_fn = lambda x,y,z : self._callback(x,y,z)
        self.callback_fn = callback_fn
    
    def start_frame_collector(self):
        self.recorded_frames = []
        self.prepare_frame_collector()
        self.approximate_time_sync = message_filters.ApproximateTimeSynchronizer(self.subscribers, 10,1)
        
        self.approximate_time_sync.registerCallback(self.callback_fn)
        print(":: Frame collector started...")
        time.sleep(1)
    
    def stop_frame_collector(self):
        self.recorded_frames = []
        for sub in self.subscribers:
            sub.unregister()
    
    def get_hparams(self):
        return copy.deepcopy(self._hp)
    
    def _default_hparams(self):
        default_dict = {
            'env_type' : 0,
            'use_rs' : False,
            'rs_to_ee_path' : '/scratch0/zhang401local/plant_data_kinect_2/realsense_ee_orig.tf',
            'icp_yaml_file' : '/scratch0/zhang401local/plant_data_kinect_2/may_18_icp_test.yaml',
            'intrinsics_file_rs_path' : '/scratch0/zhang401local/plant_data_kinect_2/realsense_intrinsics.intr',
            'crop_scene_with_aabb' : False,
            'generate_height_image_and_control_background' : False,
            'color_topic' : '/camera_mc/rgb/image_raw',
            'depth_topic' : '/camera_mc/depth_to_rgb/image_raw',
            'robot_state_topic' : '/robot_state_publisher_node_1/robot_state',
            'initial_joints' : '/home/zhang401local/nov_2_cropped_1_initial_2.npy',
            'intrinsics_file_path': '/home/zhang401local/camera-calibration/calib/azure_kinect.intr',
            'plant_angle': 0.0,
            'plant_ee_align_transformation_path' : '',
            'plant_upright_transformation_path' : '',
            'plant_upright_transformation_path_height' : '/scratch0/zhang401local/plant_data/jan_11_2022_plant_upright_with_0.001.tf',
            'extrinsics_file_path' : '',
            'height_map_dimensions' : [200, 150],
            'heightmap_resolution' : 0.005,
            'workspace_z_buffer' : 0.12,
            'action_length_unit' : 0.15,
            'plant_space_pcd' : '/scratch0/zhang401local/plant_data/cropped_jan_11.ply',
            'workspace_pcd' : '/scratch0/zhang401local/plant_data/cropped_jan_11_actionspace.ply',
            'global_workspace_limits' : np.array([[-0.31808337,  0.69382244],
                    [-0.87492469,  0.06351901],
                    [-0.64      , -0.53016218]]),
            'four_corners_global_in_image' : np.array([[ 488, 1137],
                        [ 196,   -3],
                        [1444, 1013],
                        [1760,   87]]),
            'base_height' : -0.64, #ground_height
            'grabber_width' : 0.09,
            'plant_height_max' : 1.0,  
            'plant_depth_max' : 1.5, 
            'too_far_height' : 0.002,
        }
        return default_dict
    
    def _pt_coord_with_grabber(self, pt, angle=None, arm_length=0.5):
        pt_x, pt_y, pt_z = pt 
        return np.array([pt_x, pt_y, pt_z+arm_length])
        x_dec = arm_length * np.sin(np.abs(angle))
        y_dec = arm_length * np.cos(np.abs(angle))
        return np.array([pt_x - x_dec, pt_y - y_dec, pt_z])
    
    def step(self, action):
        """
        Applies the action and steps simulation
        :param action: action at time-step
        :return: obs dict where:
                  -each key is an observation at that step
                  -keys are constant across entire datastep (e.x. every-timestep has 'state' key)
                  -keys corresponding to numpy arrays should have constant shape every timestep (for caching)
                  -images should be placed in the 'images' key in a (ncam, ...) array
        """
        pt_x, pt_y, pt_z, theta, length = action[:5]
        dx = np.cos(theta) * length
        dy = np.sin(theta) * length
        start_pt_3d = Point(np.asarray([pt_x,pt_y,pt_z]), frame="azure_kinect_overhead_upright")
        start_pt_3d_world_frame = self.T_sampled_action_space_to_world * start_pt_3d
        p1 = start_pt_3d_world_frame.data

        end_pt_3d = Point(np.asarray([pt_x+dx,pt_y+dy,pt_z]), frame="azure_kinect_overhead_upright")
        end_pt_3d_in_world_frame = self.T_sampled_action_space_to_world * end_pt_3d
        p2 = end_pt_3d_in_world_frame.data
        action_information = {
            "pix_x" : action[0],
            "pix_y" : action[1],
            "angle" : theta,
            "length" : length,
            "start_pt" : start_pt_3d.data,
            "end_pt" : end_pt_3d.data,
            "dx" : dx,
            "dy": dy,
            "start_pt_world" : start_pt_3d_world_frame.data,
            "end_pt_world" : end_pt_3d_in_world_frame.data,
        }
        if len(action) > 5:
            action_information['z_level'] = int(action[5])
        
        important_frames = [] 
        # time.sleep(1)
        start_frame = len(self.recorded_frames)
        # create a pcd and height map
        print(":: Check start_frame =", start_frame)
        # _, rgb_image, depth_image = self.recorded_frames[max(0, start_frame-1)]
        # color_pts, surface_pts = plant_utils.from_rgb_depth_image_to_color_surface_points(rgb_image, depth_image, lambda x : self.postprocess_raw_pointcloud(x), self.intrinsic_o3d)
        # self.fa.reset_joints(duration=5)
        jt1 = FC.HOME_JOINTS.copy()
        # jt1[1] = -np.pi / 2.3
        # jt1[1] = np.radians(-25)
        jt1[3] = np.radians(-135)
        self.fa.goto_joints(jt1, ignore_virtual_walls=True)
        
        start_point_ee_translation = self._pt_coord_with_grabber(p1, self.plant_angle_radians, arm_length=0.37) 
        pose_2 = RigidTransform(rotation=self.T_alignplant.rotation, translation=start_point_ee_translation, from_frame='franka_tool', to_frame='world')
        
        self.fa.goto_pose(pose_2, duration=4)
        important_frames.append(len(self.recorded_frames))

        start_point_ee_translation = self._pt_coord_with_grabber(p1, self.plant_angle_radians, arm_length=0.26) 
        pose_2_2 = RigidTransform(rotation=self.T_alignplant.rotation, translation=start_point_ee_translation, from_frame='franka_tool', to_frame='world')
        self.fa.goto_pose(pose_2_2, duration=3, use_impedance=True, force_thresholds=[10]*6)
        important_frames.append(len(self.recorded_frames))

        start_pose = self.fa.get_pose()
        if np.linalg.norm(start_pose.translation - pose_2_2.translation) > 0.04:
            self.fa.goto_pose(pose_2, duration=5)
            post_action_frame = important_frames[-1]
            afteraction_rgb_image = self.recorded_frames[post_action_frame-1][1]
            afteraction_depth_image_avg = self.get_averaged_depth_images(post_action_frame-5, 2)
            return_infos = {
                'start_frame' : start_frame,
                'action_frame_infos' : important_frames,
                'success' : False,
                'afteraction_infos' : [afteraction_rgb_image, afteraction_depth_image_avg],
            }
            if self._hp['use_rs']:
                afteraction_rgb_image_rs = self.recorded_frames[post_action_frame-1][3]
                afteraction_depth_image_rs_avg = self.get_averaged_depth_images(post_action_frame-5, 4)
                ee_pose = self.recorded_frames[post_action_frame-1][5]
                return_infos['afteraction_infos_rs'] = (afteraction_rgb_image_rs, afteraction_depth_image_rs_avg, ee_pose)

            return_infos.update(action_information)
            
            collected_frames = [(frame_num, self.recorded_frames[frame_num]) for frame_num in range(important_frames[-1])]
            return return_infos, collected_frames

        
        dist = np.linalg.norm(p1-p2)

        end_point_ee_translation = self._pt_coord_with_grabber(p2, self.plant_angle_radians, arm_length=0.26) 
        pose_3 = RigidTransform(rotation=self.T_alignplant.rotation, translation=end_point_ee_translation, from_frame='franka_tool', to_frame='world')
        self.fa.goto_pose(pose_3, duration=get_duration(dist), cartesian_impedances=[300,300,300,20,20,20], force_thresholds=[10]*6)
        time.sleep(3)
        afteraction_frame = len(self.recorded_frames)
        important_frames.append(afteraction_frame)
        
        afteraction_rgb_image = self.recorded_frames[afteraction_frame-1][1]
        afteraction_depth_image_avg = self.get_averaged_depth_images(afteraction_frame-5, 2)
        

        # afteraction_color_pts, afteraction_surface_pts = plant_utils.from_rgb_depth_image_to_color_surface_points(afteraction_rgb_image, afteraction_depth_image, lambda x : self.postprocess_raw_pointcloud(x), self.intrinsic_o3d)
        time.sleep(1)
 
        pose_4 = RigidTransform(rotation=self.T_alignplant.rotation, translation=start_point_ee_translation, from_frame='franka_tool', to_frame='world')
        # self.fa.goto_pose(pose_4, duration=get_duration(dist), cartesian_impedances=[300,300,300,20,20,20])
        important_frames.append(len(self.recorded_frames))

        pose_away_from_board = self.fa.get_pose().copy() 
        pose_away_from_board.translation = self._pt_coord_with_grabber(pose_away_from_board.translation, self.plant_angle_radians, arm_length=0.15)
        self.fa.goto_pose(pose_away_from_board, duration=3, cartesian_impedances=[300,300,300,20,20,20])
        important_frames.append(len(self.recorded_frames))
        return_infos = {
            'start_frame' : start_frame,
            'action_frame_infos' : important_frames,
            # 'start_infos' : [rgb_image, depth_image],
            'afteraction_infos' : [afteraction_rgb_image, afteraction_depth_image_avg],
            'success' : True,
        }
        if self._hp['use_rs']:
            afteraction_rgb_image_rs = self.recorded_frames[afteraction_frame-1][3]
            afteraction_depth_image_rs_avg = self.get_averaged_depth_images(afteraction_frame-5, 4)
            ee_pose = self.recorded_frames[afteraction_frame-1][5]
            return_infos['afteraction_infos_rs'] = (afteraction_rgb_image_rs, afteraction_depth_image_rs_avg, ee_pose)

        
        return_infos.update(action_information)
        collected_frames = [(frame_num, self.recorded_frames[frame_num]) for frame_num in range(important_frames[-1])]
        return return_infos, collected_frames
    
    def get_azure_kinect_rgb_image(self):
        rgb_image_msg = rospy.wait_for_message(self._hp['color_topic'], Image)
        try:
            rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_image_msg, "rgb8")
        except CvBridgeError as e:
            print(e)
        
        return np.ascontiguousarray(rgb_cv_image).astype(np.uint8)

    def get_azure_kinect_depth_image(self):
        depth_image_msg = rospy.wait_for_message(self._hp['depth_topic'], Image)
        try:
            depth_cv_image = self.cv_bridge.imgmsg_to_cv2(depth_image_msg)
        except CvBridgeError as e:
            print(e)
        
        return np.ascontiguousarray(depth_cv_image).astype(np.uint16)
    
    def create_pointcloud(self, rgb, depth, is_realsense=False):
        if is_realsense and not self._hp['use_rs']:
            raise ValueError(":: In the config file for the environment, please set use_rs to be True!!")
        color_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
        if is_realsense:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsic_o3d_rs)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsic_o3d)
        return pcd

    def align_rs_to_kinect(self, pcd_rs, ee_pose):
        T_ee_world = RigidTransform(rotation = np.asarray(ee_pose[:3,:3]), translation=ee_pose[:3,-1], from_frame="franka_tool", to_frame="world")
        T_rs_world = T_ee_world * self.T_rs_ee

        pcd_rs.transform(T_rs_world.matrix)

        return pcd_rs

    def get_height_image(self, pcd):
        pcd_pts = PointCloud(np.asarray(pcd.points).T, frame='azure_kinect_overhead')
        x_coords, y_coords = self.intrinsic_iam.project(pcd_pts).data
        pcd.transform(self.T_flip.matrix)
        pcd.transform(self.T_camera_cameraupright_height.matrix)
        pcd_points_in_frontal_parallel_frame = np.asarray(pcd.points)
        pcd_points_in_frontal_parallel_frame[:,2] -= self._hp['base_height']
        new_height_image = np.zeros((int(self.intrinsic_iam.height),int(self.intrinsic_iam.width))) # 
        new_height_image[y_coords, x_coords] = pcd_points_in_frontal_parallel_frame[:,2]
        return new_height_image
    
    def generate_depth_and_height_image(self, pcd, pcd_rs_world = None):
        return plant_utils.generate_depth_and_height_image_control_background(
            pcd, 
            self.intrinsic_iam, lambda x : self.postprocess_raw_pointcloud(pcd, crop=False), 
            self._hp['base_height'], 
            self._hp['plant_height_max'],  
            self._hp['plant_depth_max'], 
            self._hp['too_far_height'], 
            pcd_rs_world = pcd_rs_world,
        )

    def postprocess_raw_pointcloud(self, pcd, crop = True):
        pcd2 = copy.deepcopy(pcd)
        if crop:
            pcd2 = pcd2.crop(self.plant_space_obb)
        pcd2.transform(self.T_flip.matrix)
        pcd2.transform(self.T_camera_cameraupright.matrix)
        return pcd2

    def get_averaged_depth_images(self, frame_num, stack_idx):
        length_of_frames = len(self.recorded_frames)
        stack_of_depth_images = [x[stack_idx] for x in self.recorded_frames[:length_of_frames]]
        
        depth = plant_utils.get_averaged_depth_images_from_stack_of_depth_images(stack_of_depth_images, frame_num, num_frames_to_average=5) 
        return depth   
    
    def current_obs(self):
        """
        :return: Current environment observation dict
        """
        print(":: Getting observation from camera")

        if len(self.recorded_frames) == 0:
            raise ValueError(":: Length of recorded_frames is 0! Need a couple of frames in order to average depth.")
            # start_frame = 0
            # rgb = self.get_azure_kinect_rgb_image()
            # depth = self.get_azure_kinect_depth_image()
        else:
            length_of_frames = len(self.recorded_frames)
            start_frame = min(max(0, length_of_frames - 5), length_of_frames-1)
            rgb = self.recorded_frames[start_frame][1]
            depth = self.get_averaged_depth_images(start_frame, 2)
        
        
        pcd = self.create_pointcloud(rgb, depth)
        if self._hp['generate_height_image_and_control_background']:
            _,height = self.generate_depth_and_height_image(pcd, pcd_rs_world = None)
        else:
            height = self.get_height_image(pcd)
        height_f = plant_utils.median_blur_height_map(height)
        obs = {
            "start_frame" : start_frame,
            "rgb" : rgb,
            "depth" : depth,
            "height" : height,
            'height_f' : height_f,
        }
        # import pickle
        # with open("/scratch0/zhang401local/plant_data_kinect_2/test_rebuttal_obs.pkl", "wb+") as f:
        #     pickle.dump(obs, f)
        # o3d.io.write_point_cloud("/scratch0/zhang401local/plant_data_kinect_2/test_rebuttal_obs.pcd", pcd)
        return obs
    
    def icp(self, source_info, target_info):
        config = self._hp['icp_config']
        source = source_info['pcd']
        source_to_cmp = source_info['pcd_full']
        target = copy.deepcopy(target_info["pcd"])
        source_down, source_fpfh = plant_utils.preprocess_point_cloud(source, config)
        target_down, target_fpfh = plant_utils.preprocess_point_cloud(target, config)
        result_ransac = plant_utils.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, config['voxel_size'])
        trans_init = result_ransac.transformation

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_down, target_down, config["icp_threshold"], trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        T_source_target = RigidTransform(rotation=reg_p2p.transformation[:3,:3],translation=reg_p2p.transformation[:-1,-1], from_frame='source', to_frame='target')

        if 'pcd_rs' in target_info:
            pcd_rs_world = self.align_rs_to_kinect(target_info['pcd_rs'], target_info['ee_pose'])
        else:
            pcd_rs_world = None

        target_to_cmp = target_info['pcd_full']

        pcd1 = source_to_cmp # camera frame
        if pcd_rs_world:
            pcd2 = copy.deepcopy(target_to_cmp).transform(self.T_camera_world.matrix) + pcd_rs_world
            pcd2.transform(self.T_camera_world.inverse().matrix)
        else:
            pcd2 = copy.deepcopy(target_to_cmp)
        pcd3 = copy.deepcopy(pcd2)
        pcd2.transform(T_source_target.inverse().matrix)
        # import pdb;pdb.set_trace()
        return pcd1, pcd2, pcd3, pcd_rs_world, T_source_target

    def valid_rollout(self):
        """
        Checks if the environment is currently in a valid state
        Common invalid states include:
            - object falling out of bin
            - mujoco error during rollout
        :return: bool value that is False if rollout isn't valid
        """
        return 'y' in input('Valid rollout? (y if yes): ')

    def goal_reached(self):
        """
        Checks if the environment hit a goal (if environment has goals)
            - e.x. if goal is to lift object should return true if object lifted by gripper
        :return: whether or not environment reached goal state
        """
        return 'y' in input('Goal reached? (y if yes): ')

    def has_goal(self):
        """
        :return: Whether or not environment has a goal
        """
        return True

    @property
    def adim(self):
        """
        :return: Environment's action dimension
        """
        return 4

    @property
    def sdim(self):
        """
        :return: Environment's state dimension
        """
        raise NotImplementedError

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def default_ncam():
        
        return 1
