# Developed by Jingyi Xu, Junyi Ma, Zijie Zhou
# Brief: split query and database for NUSC-PR
# NUSC-PR is proposed in the paper: Explicit Interaction for Fusion-Based Place Recognition

import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
import pickle
import yaml


def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def main(config):

    random.seed(1)
    prev_root = config['data_root']['save_root_basic_infos']
    save_root = config['data_root']['save_root_splitted_data']

    check_dir(save_root)
    locations = config['locations_abbr']

    for location in locations:

        infos_path = os.path.join(prev_root, 'nuscenes_infos-'+location+'.pkl')
        
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)

        poses = []
        timestamps = []
        sample_tokens = []
        sample_tokens_db = []
        for i, info in enumerate(infos):
            pose = info['lidar_infos']['LIDAR_TOP']['ego_pose']['translation']
            poses.append(pose[:2])
            timestamp = info['timestamp']
            timestamps.append(timestamp)
            sample_token = info['sample_token']
            sample_tokens.append(sample_token)
        poses = np.array(poses, dtype=np.float32)
        timestamps = np.array(timestamps, dtype=np.float32).reshape(-1, 1)
        print('total frames for '+location+':', i)

        ############################################################
        print('==> generating database for '+location)
        distance_interval = config['distance_interval']

        poses = np.concatenate(
            (np.arange(len(poses), dtype=np.int32).reshape(-1, 1), np.array(poses)),
            axis=1).astype(np.float32)
        
        poses_db = poses[0, :].reshape(1, -1)
        sample_tokens_db.append(sample_tokens[0])
        for i in range(1, poses.shape[0]):
            knn = NearestNeighbors(n_neighbors=1)
            knn.fit(poses_db[:, 1:3])
            dis, index = knn.kneighbors(poses[i, 1:3].reshape(1, -1), 1, return_distance=True)
            if dis > distance_interval:
                poses_db = np.concatenate((poses_db, poses[i, :].reshape(1, -1)), axis=0)
                sample_tokens_db.append(sample_tokens[i])

        print('number of database frames in '+location+': ', poses_db.shape[0])
        print('finding corresponding tokens: ', len(sample_tokens_db))

        ############################################################
        print('==> generating query indices for '+location)
        timestamps = np.array(timestamps - min(timestamps)) / (3600 * 24 * 1e6)
        
        SEPERATE_TH = 200
        if location == 'bs':
            SEPERATE_TH = config['date_threshold_bs']
        elif location == 'sq':
            SEPERATE_TH = config['date_threshold_sq']
        elif location == 'son':
            SEPERATE_TH = config['date_threshold_son']
        elif location == 'shv':
            SEPERATE_TH = config['date_threshold_shv']

        train_indices, _ = np.where(timestamps < SEPERATE_TH)
        testval_indices, _ = np.where(timestamps >= SEPERATE_TH)

        database_indices = poses_db[:, 0].astype(int)
        train_query_indices = list(set(train_indices) - set(database_indices))
        testval_query_indices = list(set(testval_indices) - set(database_indices))
        val_ratio = config['val_ratio']
        val_query_indices = random.sample(testval_query_indices, int(len(testval_query_indices) * val_ratio))
        test_query_indices = list(set(testval_query_indices) - set(val_query_indices))
        
        poses_train_query = poses[train_query_indices]
        poses_test_query = poses[test_query_indices]
        poses_val_query = poses[val_query_indices]

        tokens_train_query = np.array(sample_tokens)[train_query_indices]
        tokens_test_query = np.array(sample_tokens)[test_query_indices]
        tokens_val_query = np.array(sample_tokens)[val_query_indices]

        print('the number of train query frames in '+location+': ', tokens_train_query.shape[0])
        print('the number of test query frames in '+location+': ', tokens_test_query.shape[0])
        print('the number of val query frames in '+location+': ', tokens_val_query.shape[0])

        ############################################################
        print('===> saving database and queries for '+location)
        np.save(os.path.join(save_root, location+'_db.npy'), poses_db)
        np.save(os.path.join(save_root, location+'_train_query.npy'), poses_train_query)
        np.save(os.path.join(save_root, location+'_val_query.npy'), poses_val_query)
        np.save(os.path.join(save_root, location+'_test_query.npy'), poses_test_query)
        np.save(os.path.join(save_root, location+'_db_sample_token.npy'), sample_tokens_db)
        np.save(os.path.join(save_root, location+'_train_query_sample_token.npy'), tokens_train_query)
        np.save(os.path.join(save_root, location+'_val_query_sample_token.npy'), tokens_val_query)
        np.save(os.path.join(save_root, location+'_test_query_sample_token.npy'), tokens_test_query)
        np.save(os.path.join(save_root, location+'_sample_token.npy'), sample_tokens)
        print("=========================================")

    print("done!")

if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../configs/config_supervised_scheme.yml'
    config = yaml.safe_load(open(config_filename))
    # ============================================================================
    main(config)    