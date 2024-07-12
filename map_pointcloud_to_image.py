def map_pointcloud_to_image(self,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None,
                            show_panoptic: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = self.nusc.get('sample_data', camera_token)
    pointsensor = self.nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        if show_lidarseg or show_panoptic:
            gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
            assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

            assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                            'render_intensity or show_lidarseg, not both.'

        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                            'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                            'not %s!' % pointsensor['sensor_modality']

        gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
        semantic_table = getattr(self.nusc, gt_from)

        if lidarseg_preds_bin_path:
            sample_token = self.nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                    self.nusc.get(gt_from, pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None

        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(lidarseg_labels_filename,
                                                filter_lidarseg_labels,
                                                self.nusc.lidarseg_name2idx_mapping,
                                                self.nusc.colormap)
            else:
                coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                    filter_lidarseg_labels,
                                                    self.nusc.lidarseg_name2idx_mapping,
                                                    self.nusc.colormap)

        else:
            coloring = depths
            print(f'Warning: There are no lidarseg labels in {self.nusc.version}. Points will be colored according '
                    f'to distance from the ego vehicle instead.')
    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    coloring = coloring[mask]

    return points, coloring, im, depths