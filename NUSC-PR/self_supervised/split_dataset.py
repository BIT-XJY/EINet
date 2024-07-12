# Developed by Jingyi Xu, Junyi Ma, Zijie Zhou
# Brief: split query and database for NUSC-PR
# NUSC-PR is proposed in the paper: Explicit Interaction for Fusion-Based Place Recognition


import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle
from nuscenes import NuScenes
import yaml

def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def scene_index_to_sample_index(scene_index, scene_nbr_list):
    sample_index_start = 0
    for i in range(scene_index):
        sample_index_start += scene_nbr_list[i]
    sample_index_end = sample_index_start + scene_nbr_list[scene_index] - 1
    return sample_index_start, sample_index_end

def load_txt_index(txt_path):
    index_list = []
    with open(txt_path, 'r') as f:
        index_lines = f.readlines()
    for i in range(len(index_lines)):
        index_list.append(int(index_lines[i].split('\n')[0]))
    return index_list

def translation_to_info_index(t_list, translation_index, scene_nbr_list):
    # t_list为train_list或test_list
    index_list = []
    index_all = 0
    for i in t_list:
        index_all += scene_nbr_list[i]
        index_list.append(index_all - 1)
    for i in range(len(index_list)):
        if translation_index <= index_list[0]:
            t_scene_index = t_list[0]
            rest_test = translation_index
        else:
            if (translation_index <= index_list[i]) & (translation_index > index_list[i-1]):
                t_scene_index = t_list[i]
                rest_test = translation_index - index_list[i-1] - 1

    info_index = 0
    for i in range(t_scene_index):
        info_index += scene_nbr_list[i]
    info_index = info_index + rest_test
    return info_index

def split_test_for_different_loc(infos, database_list_loc, test_list_loc, scene_nbr_list, save_root, loc, positive_distance_threshold):
    database_indicies = []
    database_lidar_translation = []
    for i in range(len(database_list_loc)):
        database_scene_index = database_list_loc[i]
        sample_index_start, sample_index_end = scene_index_to_sample_index(database_scene_index, scene_nbr_list)

        for j in range(scene_nbr_list[database_scene_index]):
            sample_index = sample_index_start + j
            database_indicies.append(sample_index)
            sample_info = infos[sample_index]
            lidar_translation = sample_info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
            database_lidar_translation.append(lidar_translation)
    
    np.save(os.path.join(save_root, loc+'_db_index_in_infos.npy'), database_indicies)

    test_lidar_translation = []
    for i in range(len(test_list_loc)):
        test_scene_index = test_list_loc[i]
        sample_index_start, sample_index_end = scene_index_to_sample_index(test_scene_index, scene_nbr_list)

        for j in range(scene_nbr_list[test_scene_index]):
            sample_index = sample_index_start + j
            sample_info = infos[sample_index]
            lidar_translation = sample_info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
            test_lidar_translation.append(lidar_translation)
    
    knn = NearestNeighbors(n_jobs=-1)
    database_lidar_translation = np.array(database_lidar_translation)
    test_lidar_translation = np.array(test_lidar_translation)
    database = np.ascontiguousarray(database_lidar_translation)
    queries = np.ascontiguousarray(test_lidar_translation)
    knn.fit(database)
    dis, gt_samples = knn.radius_neighbors(queries, radius=positive_distance_threshold, return_distance=True)

    # convert to index in infos
    gt_mapping = dict()
    for i in range(len(gt_samples)):
        gt_index_list = []
        test_query = translation_to_info_index(test_list_loc, i, scene_nbr_list)
        for j in range(len(gt_samples[i])):
            translation_index = gt_samples[i][j]
            gt_index_list.append(translation_to_info_index(database_list_loc, translation_index, scene_nbr_list))
        gt_mapping[test_query] = gt_index_list
    
    with open(os.path.join(save_root, loc+'_test_query_gt_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(gt_mapping, f)


def main(config):

    nusc_root = config["data_root"]["nusc_root"]
    save_root = config["data_root"]["save_root_splitted_data"]
    check_dir(save_root)
    infos_root = config['data_root']['save_root_basic_infos']
    num_pos = config["num_pos"]
    num_neg = config["num_neg"]
    positive_distance_threshold = config["positive_distance_threshold"]
    negative_time_threshold = config["negative_time_threshold"]
    use_date_date_threshold = config["use_date_date_threshold"]
    date_threshold = config["date_threshold"]
    locations_ = config['locations_abbr']

    infos_root = os.path.join(infos_root, 'nuscenes_infos.pkl')
    with open(infos_root, 'rb') as f:
        infos = pickle.load(f)

    nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=True)

    scene_nbr_list = []
    old_scenes = []
    new_scenes = []
    timestamps = []

    for i in range(len(nusc_trainval.scene)):
        scene_nbr_list.append(nusc_trainval.scene[i]['nbr_samples'])
        current_scene = nusc_trainval.scene[i]
        first_sample_token = current_scene['first_sample_token']
        first_sample = nusc_trainval.get('sample', first_sample_token)
        timestamp = first_sample['timestamp']
        timestamps.append(timestamp)
    timestamps = np.array(timestamps, dtype=np.float32)
    timestamps = np.array(timestamps - min(timestamps)) / (3600 * 24 * 1e6)
    
    if use_date_date_threshold:
        for i in range(len(nusc_trainval.scene)):
            if timestamps[i] < date_threshold:
                old_scenes.append(i)
            else:
                new_scenes.append(i)
    else:
        scene_indicies = np.arange(0, len(nusc_trainval.scene),1).tolist()
        new_scenes = random.sample(scene_indicies, 150)
        old_scenes = list(set(scene_indicies) - set(new_scenes))


    print('======================')
    print('the number of scenes: ', len(nusc_trainval.scene))
    print('the number of scenes as train queries and database: ', len(old_scenes))
    print('the number of scenes as test queries: ', len(new_scenes))
    print('======================')

    # ====================generate training tuples====================
    print('splitting train data ...')
    all_query_dict = dict()
    bs_query_dict = dict()
    son_query_dict = dict()
    sq_query_dict = dict()
    shv_query_dict = dict()

    for train_scene_index in old_scenes:
        sample_index_start, sample_index_end = scene_index_to_sample_index(train_scene_index, scene_nbr_list)
        list_sample = range(sample_index_start, sample_index_end+1)
        scene = nusc_trainval.scene[train_scene_index]
        loc_scene = nusc_trainval.get('log', scene['log_token'])['location']
        for j in range(scene_nbr_list[train_scene_index]):
            query_index = list_sample[j]
            
            if j < num_pos+num_neg+negative_time_threshold:
                continue 
            else:
                pos_neg_dict = dict()
                pos_index = []
                for pos in range(num_pos):
                    pos_index.append(list_sample[j-pos-1])
                pos_neg_dict['pos'] = pos_index

                potential_neg_list = []
                for i in range(0, j-num_pos-negative_time_threshold):
                    potential_neg_list.append(list_sample[i])

                neg_index = sorted(random.sample(potential_neg_list, num_neg))
                pos_neg_dict['neg'] = neg_index
                pos_neg_dict['loc'] = loc_scene
                all_query_dict[query_index] = pos_neg_dict
                if loc_scene == 'singapore-onenorth':
                    son_query_dict[query_index] = pos_neg_dict
                elif loc_scene == 'singapore-hollandvillage':
                    shv_query_dict[query_index] = pos_neg_dict
                elif loc_scene == 'singapore-queenstown':
                    sq_query_dict[query_index] = pos_neg_dict
                elif loc_scene == 'boston-seaport':
                    bs_query_dict[query_index] = pos_neg_dict
    
    with open(os.path.join(save_root, 'all_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(all_query_dict, f)
    with open(os.path.join(save_root, 'son_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(son_query_dict, f)
    with open(os.path.join(save_root, 'shv_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(shv_query_dict, f)
    with open(os.path.join(save_root, 'sq_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(sq_query_dict, f)
    with open(os.path.join(save_root, 'bs_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
        pickle.dump(bs_query_dict, f)
    print('======================')
    
    # ====================generate test tuples====================
    # We use all the samples in the old scenes as database
    # and use the samples in the new scenes as query
    print('splitting test data ...')

    son_test_list = []
    shv_test_list = []
    sq_test_list = []
    bs_test_list = []
    for i in new_scenes:
        scene = nusc_trainval.scene[i]
        loc_scene = nusc_trainval.get('log', scene['log_token'])['location']
        if loc_scene == 'singapore-onenorth':
            son_test_list.append(i)
        elif loc_scene == 'singapore-hollandvillage':
            shv_test_list.append(i)
        elif loc_scene == 'singapore-queenstown':
            sq_test_list.append(i)
        elif loc_scene == 'boston-seaport':
            bs_test_list.append(i)

    son_database_list = []
    shv_database_list = []
    sq_database_list = []
    bs_database_list = []
    for i in old_scenes:
        scene = nusc_trainval.scene[i]
        loc_scene = nusc_trainval.get('log', scene['log_token'])['location']
        if loc_scene == 'singapore-onenorth':
            son_database_list.append(i)
        elif loc_scene == 'singapore-hollandvillage':
            shv_database_list.append(i)
        elif loc_scene == 'singapore-queenstown':
            sq_database_list.append(i)
        elif loc_scene == 'boston-seaport':
            bs_database_list.append(i)
    
    # print("son")
    # print(len(son_test_list))
    # print(len(son_database_list))
    # print("shv")
    # print(len(shv_test_list))
    # print(len(shv_database_list))
    # print("sq")
    # print(len(sq_test_list))
    # print(len(sq_database_list))
    # print("bs")
    # print(len(bs_test_list))
    # print(len(bs_database_list))

    split_test_for_different_loc(infos, son_database_list, son_test_list, scene_nbr_list, save_root, 'son', positive_distance_threshold)
    split_test_for_different_loc(infos, shv_database_list, shv_test_list, scene_nbr_list, save_root, 'shv', positive_distance_threshold)    
    split_test_for_different_loc(infos, sq_database_list, sq_test_list, scene_nbr_list, save_root, 'sq', positive_distance_threshold)
    split_test_for_different_loc(infos, bs_database_list, bs_test_list, scene_nbr_list, save_root, 'bs', positive_distance_threshold)

    print("done!")

if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../configs/config_self_supervised_scheme.yml'
    config = yaml.safe_load(open(config_filename))
    # ============================================================================
    main(config)