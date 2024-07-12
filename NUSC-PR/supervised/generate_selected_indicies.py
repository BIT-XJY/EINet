# Developed by Jingyi Xu, Junyi Ma, Zijie Zhou
# Brief: convert tokens to indices in basic infos of NUSC-PR for possible use
# NUSC-PR is proposed in the paper: Explicit Interaction for Fusion-Based Place Recognition

import pickle
import numpy as np
import os
import yaml

def check_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def main(config):
    prev_root0 = config['data_root']['save_root_basic_infos']
    with open(os.path.join(prev_root0, 'nuscenes_infos.pkl'), 'rb') as f:
        infos = pickle.load(f)
    save_root = config['data_root']['save_root_selected_indices']
    check_dir(save_root)
    prev_root1 = config['data_root']['save_root_splitted_data']
    prev_root2 = config['data_root']['save_root_select_by_dis']
    locations = config['locations_abbr']


    for location in locations:
        with open(os.path.join(prev_root2, location+'_train_query_pos_neg_tokens.pkl'), 'rb') as f:
            query_pos_neg_dict = pickle.load(f)
        with open(os.path.join(prev_root2, location+'_test_query_gt_tokens.pkl'), 'rb') as f:
            gt_mapping = pickle.load(f)

        # sample_token -> id
        infos_new = {}
        for i in range(len(infos)):
            sample_token = infos[i]['sample_token']
            infos_new[sample_token] = int(i)

    ############################################################
        print("generating database indices in " + location)
        db_index_in_infos = []
        sample_tokens_db = np.load(os.path.join(prev_root1, location+'_db_sample_token.npy'))
        for i in range(len(sample_tokens_db)):
            db_token_ = sample_tokens_db[i]
            db_index = infos_new[db_token_]
            db_index_in_infos.append(db_index)

        np.save(os.path.join(save_root, location+'_db_index_in_infos.npy'), db_index_in_infos)

    ############################################################
        print("generating test-query-gt mapping indices in " + location)
        test_dict = {}
        for key, value in gt_mapping.items():
            query_sample_token = key
            test_query_index = infos_new[query_sample_token]
            
            gt_index_list = []
            for j in range(len(value['gt'])):
                gt_sample_token = value['gt'][j]
                gt_idex = infos_new[gt_sample_token]
                gt_index_list.append(gt_idex)
            test_dict[test_query_index] = gt_index_list

        with open(os.path.join(save_root, location+'_test_query_gt_index_in_infos.pkl'), 'wb') as f:
            pickle.dump(test_dict, f)

    ############################################################
        print("generating train-query-pos-neg mapping indices in " + location)
        train_dict = {}
        for key, value in query_pos_neg_dict.items():
            query_sample_token = key
            train_query_index = infos_new[query_sample_token]
            
            pos_index_list = []
            neg_index_list = []
            pos_neg_dict = {}
            for j in range(len(value['pos'])):
                pos_sample_token = value['pos'][j]
                pos_idex = infos_new[pos_sample_token]
                pos_index_list.append(pos_idex)
            pos_neg_dict['pos'] = pos_index_list

            for j in range(len(value['neg'])):
                neg_sample_token = value['neg'][j]
                neg_idex = infos_new[neg_sample_token]
                neg_index_list.append(neg_idex)
            pos_neg_dict['neg'] = neg_index_list

            train_dict[train_query_index] = pos_neg_dict

        with open(os.path.join(save_root, location+'_train_query_pos_neg_index_in_infos.pkl'), 'wb') as f:
            pickle.dump(train_dict, f)

        print("=========================================")

    print("done!")

if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../configs/config_supervised_scheme.yml'
    config = yaml.safe_load(open(config_filename))
    # ============================================================================
    main(config)    