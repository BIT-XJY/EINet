# EINet
The official code and benchmark for our paper: [**Explicit Interaction for Fusion-Based Place Recognition**](https://arxiv.org/abs/2402.17264).

This work has been accepted by IROS 2024 :tada:

[Jingyi Xu](https://github.com/BIT-XJY), [Junyi Ma](https://github.com/BIT-MJY),  [Qi Wu](https://github.com/Gatsby23), [Zijie Zhou](https://github.com/ZhouZijie77), [Yue Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=N543LSoAAAAJ), [Xieyuanli Chen](https://github.com/Chen-Xieyuanli), Wenxian Yu, Ling Pei*.

![image](https://github.com/BIT-XJY/EINet/assets/83287843/c560b31c-86d0-4ec8-a1b0-c6b8da79db0f)

## Installation

We follow the installation instructions of our codebase [LCPR](https://github.com/ZhouZijie77/LCPR), which are also posted here.
</summary>

* Create a conda virtual environment and activate it
```bash
git clone git@github.com:BIT-XJY/EINet.git
cd EINet
conda create -n EINet python=3.8
conda activate EINet
```
* Install other dependencies
```bash
pip install -r requirements.txt
```

## Data Download
- Please download the offical [nuScenes dataset](https://www.nuscenes.org/nuscenes#download), and link it to the data folder.
- [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl), and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) are also provided by the previous work. We also use it to train the EINet.

Note that the download data structure should be like: 
```
nuscenes
├─ raw_data
│    ├─ maps
│    │    ├─ ...
│    ├─ samples
│    │    ├─ CAM_BACK
│    │    ├─ CAM_BACK_LEFT
│    │    ├─ CAM_BACK_RIGHT
│    │    ├─ CAM_FRONT
│    │    ├─ CAM_FRONT_LEFT
│    │    ├─ CAM_FRONT_RIGHT
│    │    ├─ LIDAR_TOP
│    │    ├─ RADAR_BACK_LEFT
│    │    ├─ RADAR_BACK_RIGHT
│    │    ├─ RADAR_FRONT
│    │    ├─ RADAR_FRONT_LEFT
│    │    ├─ RADAR_FRONT_RIGHT
│    ├─ sweeps
│    │    ├─ CAM_BACK
│    │    ├─ CAM_BACK_LEFT
│    │    ├─ CAM_BACK_RIGHT
│    │    ├─ CAM_FRONT
│    │    ├─ CAM_FRONT_LEFT
│    │    ├─ CAM_FRONT_RIGHT
│    │    ├─ LIDAR_TOP
│    │    ├─ RADAR_BACK_LEFT
│    │    ├─ RADAR_BACK_RIGHT
│    │    ├─ RADAR_FRONT
│    │    ├─ RADAR_FRONT_LEFT
│    │    ├─ RADAR_FRONT_RIGHT
│    ├─ v1.0-test
│    │    ├─ attribute.json
│    │    ├─ calibrated_sensor.json
│    │    ├─ ...
│    ├─ v1.0-traninval
│    │    ├─ attribute.json
│    │    ├─ calibrated_sensor.json
│    │    ├─ ...
```

## NUSC-PR
We propose the NUSC-PR benchmark to split nuScenes datasets with self-supervised and supervised learning schemes.

### Self-supervised Data Preparation
- Extract basic information from nuScenes datasets, and split query and database for NUSC-PR.
```bash
cd NUSC-PR
cd self_supervised
python generate_basic_infos.py
python split_dataset.py
cd ..
```

- The data structure with a self-supervised learning scheme should be like:
```
self_supervised_data
├─ generate_basic_infos
│    ├─ nuscenes_infos-bs.pkl
│    ├─ nuscenes_infos-shv.pkl
│    ├─ nuscenes_infos-son.pkl
│    ├─ nuscenes_infos-sq.pkl
│    ├─ nuscenes_infos.pkl
├─ split_dataset
│    ├─ all_train_query_pos_neg_index_in_infos.pkl
│    ├─ bs_db_index_in_infos.npy
│    ├─ bs_test_query_gt_index_in_infos.pkl
│    ├─ bs_train_query_pos_neg_index_in_infos.pkl
│    ├─ shv_db_index_in_infos.npy
│    ├─ shv_test_query_gt_index_in_infos.pkl
│    ├─ shv_train_query_pos_neg_index_in_infos.pkl
│    ├─ son_db_index_in_infos.npy
│    ├─ son_test_query_gt_index_in_infos.pkl
│    ├─ son_train_query_pos_neg_index_in_infos.pkl
│    ├─ sq_db_index_in_infos.npy
│    ├─ sq_test_query_gt_index_in_infos.pkl
│    ├─ sq_train_query_pos_neg_index_in_infos.pkl
```


### Supervised Data Preparation
- Extract basic information from nuScenes datasets, and split query and database for NUSC-PR.
```bash
cd supervised
python generate_basic_infos.py
python split_dataset.py
python select_pos_neg_samples_by_dis.py
python generate_selected_indicies.py
cd ..
cd ..
```

- The data structure with a supervised learning scheme should be like:
```
supervised_data
├─ generate_basic_infos
│    ├─ nuscenes_infos-bs.pkl
│    ├─ nuscenes_infos-shv.pkl
│    ├─ nuscenes_infos-son.pkl
│    ├─ nuscenes_infos-sq.pkl
│    ├─ nuscenes_infos.pkl
├─ generate_selected_indicies
│    ├─ bs_db_index_in_infos.npy
│    ├─ bs_test_query_gt_index_in_infos.pkl
│    ├─ bs_train_query_pos_neg_index_in_infos.pkl
│    ├─ shv_db_index_in_infos.npy
│    ├─ shv_test_query_gt_index_in_infos.pkl
│    ├─ shv_train_query_pos_neg_index_in_infos.pkl
│    ├─ son_db_index_in_infos.npy
│    ├─ son_test_query_gt_index_in_infos.pkl
│    ├─ son_train_query_pos_neg_index_in_infos.pkl
│    ├─ sq_db_index_in_infos.npy
│    ├─ sq_test_query_gt_index_in_infos.pkl
│    ├─ sq_train_query_pos_neg_index_in_infos.pkl
├─ select_pos_neg_samples_by_dis
│    ├─ bs_test_query_gt_tokens.pkl
│    ├─ bs_train_query_pos_neg_tokens.pkl
│    ├─ shv_test_query_gt_tokens.pkl
│    ├─ shv_train_query_pos_neg_tokens.pkl
│    ├─ son_test_query_gt_tokens.pkl
│    ├─ son_train_query_pos_neg_tokens.pkl
│    ├─ sq_test_query_gt_tokens.pkl
│    ├─ sq_train_query_pos_neg_tokens.pkl
├─ split_dataset
│    ├─ bs_db_sample_token.npy
│    ├─ bs_db.npy
│    ├─ bs_sample_token.npy
│    ├─ bs_test_query_sample_token.npy
│    ├─ bs_test_query.npy
│    ├─ bs_train_query_sample_token.npy
│    ├─ bs_train_query.npy
│    ├─ bs_val_query_sample_token.npy
│    ├─ bs_val_query.npy
│    ├─ shv_db_sample_token.npy
│    ├─ shv_db.npy
│    ├─ shv_sample_token.npy
│    ├─ shv_test_query_sample_token.npy
│    ├─ shv_test_query.npy
│    ├─ shv_train_query_sample_token.npy
│    ├─ shv_train_query.npy
│    ├─ shv_val_query_sample_token.npy
│    ├─ shv_val_query.npy
│    ├─ son_db_sample_token.npy
│    ├─ son_db.npy
│    ├─ son_sample_token.npy
│    ├─ son_test_query_sample_token.npy
│    ├─ son_test_query.npy
│    ├─ son_train_query_sample_token.npy
│    ├─ son_train_query.npy
│    ├─ son_val_query_sample_token.npy
│    ├─ son_val_query.npy
│    ├─ sq_db_sample_token.npy
│    ├─ sq_db.npy
│    ├─ sq_sample_token.npy
│    ├─ sq_test_query_sample_token.npy
│    ├─ sq_test_query.npy
│    ├─ sq_train_query_sample_token.npy
│    ├─ sq_train_query.npy
│    ├─ sq_val_query_sample_token.npy
│    ├─ sq_val_query.npy
```

## TODO

- [X] Release the [paper](https://arxiv.org/abs/2402.17264)
- [X] Release the benchmark NUSC-PR code for EINet
- [ ] Release the source code for EINet
- [ ] Release our pretrained baseline model


## Acknowledgement

We thank the fantastic works [LCPR](https://github.com/ZhouZijie77/LCPR), [ManyDepth](https://github.com/nianticlabs/manydepth.git), and [AutoPlace](https://github.com/ramdrop/autoplace.git) for their pioneer code release, which provide codebase for this work.
