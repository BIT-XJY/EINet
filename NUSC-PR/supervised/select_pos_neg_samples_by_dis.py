# Developed by Jingyi Xu, Junyi Ma, Zijie Zhou
# Brief: select positve and negtive samples for queries in the supervised-learning scheme of NUSC-PR
# NUSC-PR is proposed in the paper: Explicit Interaction for Fusion-Based Place Recognition

import os
# import h5py
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import yaml

def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def main(config):

    prev_root = config['data_root']['save_root_splitted_data']  
    save_root =  config['data_root']['save_root_select_by_dis']
    check_dir(save_root)
    locations = config['locations_abbr']

    for location in locations:

        poses_db = np.load(os.path.join(prev_root, location+'_db.npy'))
        sample_tokens = np.load(os.path.join(prev_root, location+'_sample_token.npy'))

        poses_test_query = np.load(os.path.join(prev_root, location+'_test_query.npy'))
        sample_tokens_db = np.load(os.path.join(prev_root, location+'_db_sample_token.npy'))
        tokens_test_query = np.load(os.path.join(prev_root, location+'_test_query_sample_token.npy'))

        poses_train_query = np.load(os.path.join(prev_root, location+'_train_query.npy'))
        tokens_train_query = np.load(os.path.join(prev_root, location+'_train_query_sample_token.npy'))

        ############################################################
        print("generating gt samples for test queries in " + location)
        positive_distance_threshold = config['positive_distance_threshold']

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(poses_db[:, 1:])
        positives_for_test = list(knn.radius_neighbors(poses_test_query[:, 1:], radius=positive_distance_threshold, return_distance=False))

        gt_mapping = {}
        for i, posi in enumerate(positives_for_test):
            positive_sample_tokens = []

            for t in posi:
                positive_index_in_db = int(t)
                positive_pose = poses_db[positive_index_in_db]
                positive_sample_token = sample_tokens[int(positive_pose[0])]
                positive_sample_tokens.append(positive_sample_token)

            gt_dict = {}
            query_key = tokens_test_query[i]
            gt_dict['gt'] = positive_sample_tokens
            gt_mapping[query_key] = gt_dict

        with open(os.path.join(save_root, location+'_test_query_gt_tokens.pkl'), 'wb') as f:
            pickle.dump(gt_mapping, f)

        ############################################################
        print("generating positive and negative samples for train queries in " + location)
        negative_distance_threshold = config['negative_distance_threshold']
        nbr_positive_samples = config['nbr_positive_samples']
        nbr_negative_samples = config['nbr_negative_samples']

        knn = NearestNeighbors()
        knn.fit(poses_db[:, 1:])
        positives_for_train = list(knn.radius_neighbors(poses_train_query[:, 1:],
                                                                radius=positive_distance_threshold,
                                                                return_distance=False))
        query_ref_mapping = {}
        for i, posi in enumerate(positives_for_train):
            positive_sample_tokens = []
            selected_positives = np.random.choice(positives_for_train[i], nbr_positive_samples)

            for t in selected_positives:
                positive_index_in_db = int(t)
                positive_pose = poses_db[positive_index_in_db]
                positive_sample_token = sample_tokens[int(positive_pose[0])]
                positive_sample_tokens.append(positive_sample_token)

            mapping_dict = {}
            query_key = tokens_train_query[i]
            mapping_dict['pos'] = positive_sample_tokens
            query_ref_mapping[query_key] = mapping_dict

        selected_no_negtives = list(knn.radius_neighbors(poses_train_query[:, 1:], radius=negative_distance_threshold,
                                                        return_distance=False))
        for i, posi in enumerate(selected_no_negtives):

            negative_sample_tokens = []
            potential_negatives = np.setdiff1d(np.arange(poses_db.shape[0]), posi,
                                                            assume_unique=True)
            selected_negtives = np.random.choice(potential_negatives, nbr_negative_samples)
            for t in selected_negtives:
                negtive_index_in_db = int(t)
                negtive_pose = poses_db[negtive_index_in_db]
                negtive_sample_token = sample_tokens[int(negtive_pose[0])]
                negative_sample_tokens.append(negtive_sample_token)
            
            query_key = tokens_train_query[i]
            mapping_dict = query_ref_mapping[query_key]
            mapping_dict['neg'] = negative_sample_tokens
            query_ref_mapping[query_key] = mapping_dict

        with open(os.path.join(save_root, location+'_train_query_pos_neg_tokens.pkl'), 'wb') as f:
            pickle.dump(query_ref_mapping, f)

        print("=========================================")

    print("done!")

if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../configs/config_supervised_scheme.yml'
    config = yaml.safe_load(open(config_filename))
    # ============================================================================
    main(config)    