# Developed by Jingyi Xu, Junyi Ma, Zijie Zhou
# Brief: extract basic infos from nuScenes datasets
# NUSC-PR is proposed in the paper: Explicit Interaction for Fusion-Based Place Recognition

import os
import pickle
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import yaml


def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def gen_info(nusc, sample_tokens):
    # Collect sample infos into a dict
    cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    lidar_names = ['LIDAR_TOP']
    
    infos = list()
    for sample_token in tqdm(sample_tokens):
        sample = nusc.get('sample', sample_token)
        info = dict()
        cam_datas = list()
        lidar_datas = list()
        info['sample_token'] = sample_token
        info['timestamp'] = sample['timestamp']
        info['scene_token'] = sample['scene_token']
        
        cam_infos = dict()
        lidar_infos = dict()
        for cam_name in cam_names:
            cam_data = nusc.get('sample_data', sample['data'][cam_name])
            cam_datas.append(cam_data)
            cam_info = dict()
            cam_info['sample_token'] = sample['data'][cam_name]
            cam_info['ego_pose'] = nusc.get('ego_pose', cam_data['ego_pose_token'])
            cam_info['timestample'] = cam_data['timestamp']
            cam_info['filename'] = cam_data['filename']
            cam_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_infos[cam_name] = cam_info
        for lidar_name in lidar_names:
            lidar_data = nusc.get('sample_data',
                                  sample['data'][lidar_name])
            lidar_datas.append(lidar_data)
            lidar_info = dict()
            lidar_info['sample_token'] = sample['data'][lidar_name]
            lidar_info['ego_pose'] = nusc.get(
                'ego_pose', lidar_data['ego_pose_token'])
            lidar_info['timestamp'] = lidar_data['timestamp']
            lidar_info['filename'] = lidar_data['filename']
            lidar_info['calibrated_sensor'] = nusc.get(
                'calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar_infos[lidar_name] = lidar_info
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        scene = nusc.get('scene', sample['scene_token'])
        loc_scene = nusc.get('log', scene['log_token'])['location']
        info['loc'] = loc_scene
        infos.append(info)

    return infos


def get_location_sample_tokens(nusc, location):
    # Get the sample tokens of a specific location

    location_indices = get_location_indices(nusc, location)

    sample_token_list = []

    for scene_index in location_indices:
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']

        while not sample_token == '':
            sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            sample_token = sample['next']

    return sample_token_list


def get_location_indices(nusc, location):
    # Get the indices of the specific location

    location_indices = []
    for scene_index in range(len(nusc.scene)):
        scene = nusc.scene[scene_index]
        if nusc.get('log', scene['log_token'])['location'] != location:
            continue
        location_indices.append(scene_index)
    return np.array(location_indices)

def get_sample_tokens(nusc):
    # Get all sample tokens

    sample_token_list = []
    for scene_index in range(len(nusc.scene)):
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']
    
        while not sample_token == '':
            sample = nusc.get('sample', sample_token)
            sample_token_list.append(sample_token)
            sample_token = sample['next']
    return sample_token_list

def main(config):

    nusc_root = config['data_root']['nusc_root']
    save_root = config['data_root']['save_root_basic_infos']
    check_dir(save_root)

    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=True)

    locations = config['locations']
    locations_ = config['locations_abbr']

    # ====================generate location infos====================
    for idx, location in enumerate(locations):
        sample_tokens = get_location_sample_tokens(nusc_trainval, location=location)
        location_infos = gen_info(nusc_trainval, sample_tokens)
        with open(os.path.join(save_root, 'nuscenes_infos-'+locations_[idx]+'.pkl'), 'wb') as f:
            pickle.dump(location_infos, f)
    
    # ====================generate all infos====================
    sample_tokens_trainval = get_sample_tokens(nusc_trainval)
    infos = gen_info(nusc_trainval, sample_tokens_trainval)
    with open(os.path.join(save_root, 'nuscenes_infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)

    print("done!")


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../configs/config_self_supervised_scheme.yml'
    config = yaml.safe_load(open(config_filename))
    # ============================================================================
    main(config)