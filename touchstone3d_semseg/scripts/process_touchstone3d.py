""" Collect point clouds and the corresponding labels from Touchstone3D dataset and process the files.

Author: Umamaheswaran Raman Kumar & Abdur R. Fayjie, 2023
"""

import os
import glob
import numpy as np
import pandas as pd
import sys
import argparse
import pickle
import yaml
import h5py as h5
from itertools import  combinations
from pyntcloud import PyntCloud
import open3d as o3d

from utils import *
from touchstone3d import Touchstone3DDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/touchstone3d/touchstone3d_semseg/scripts/process_touchstone3d.yaml',
                        help='Config file path')
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    META_PATH = os.path.join(config['DATA']['ROOT_DIR'], 'raw', 'meta', 'touchstone3d_classnames.txt')
    DATA_PATH = os.path.join(config['DATA']['ROOT_DIR'], 'raw', 'data')
    DST_PATH = os.path.join(config['DATA']['ROOT_DIR'], 'processed')
    TRAIN_ROOM_PATH = os.path.join(DST_PATH, 'train', 'rooms')
    if not os.path.exists(TRAIN_ROOM_PATH): os.makedirs(TRAIN_ROOM_PATH)
    TRAIN_BLOCK_PATH = os.path.join(DST_PATH, 'train', 'blocks')
    if not os.path.exists(TRAIN_BLOCK_PATH): os.makedirs(TRAIN_BLOCK_PATH)
    VAL_ROOM_PATH = os.path.join(DST_PATH, 'val', 'rooms')
    if not os.path.exists(VAL_ROOM_PATH): os.makedirs(VAL_ROOM_PATH)
    VAL_BLOCK_PATH = os.path.join(DST_PATH, 'val', 'blocks')
    if not os.path.exists(VAL_BLOCK_PATH): os.makedirs(VAL_BLOCK_PATH)
    TEST_ROOM_PATH = os.path.join(DST_PATH, 'test', 'rooms')
    if not os.path.exists(TEST_ROOM_PATH): os.makedirs(TEST_ROOM_PATH)
    TEST_BLOCK_PATH = os.path.join(DST_PATH, 'test', 'blocks')
    if not os.path.exists(TEST_BLOCK_PATH): os.makedirs(TEST_BLOCK_PATH)
    DST_META_PATH = os.path.join(DST_PATH, 'meta')
    if not os.path.exists(DST_META_PATH): os.makedirs(DST_META_PATH)    
    PKL_MAP_FILE = os.path.join(DST_META_PATH, 'map.pkl')

    # Metadata file mapping
    CLASS_LABELS = list([x.rstrip().split()[0]
                   for x in open(META_PATH)])
    COLOR2LABEL = {tuple(map(int, x.rstrip().split()[1:4])): x.rstrip().split()[0]
                   for x in open(META_PATH)}
    LABEL2CLASS = {cls: i for i, cls in enumerate(CLASS_LABELS)}
    
    
    for folder in os.listdir(DATA_PATH):
        print('\nProcessing folder : ', folder)
        for file_path in glob.glob(DATA_PATH+'/'+folder+'/*.pcd'):
            in_filename = os.path.split(file_path)[1]
            print(in_filename)
            f_split = in_filename.split('.')
            f_split = [f for f in f_split if f not in ('obj', 'groundtruth', 'pcd')]
            out_filename = folder+'_floor'+f_split[0] + '_'+''.join(f_split[1:])+'.npy'
            if folder in config['TRAIN']['HOUSES']:
                room_out_filepath = os.path.join(TRAIN_ROOM_PATH, out_filename)
            elif folder in config['VAL']['HOUSES']:
                room_out_filepath = os.path.join(VAL_ROOM_PATH, out_filename)
            elif folder in config['TEST']['HOUSES']:
                room_out_filepath = os.path.join(TEST_ROOM_PATH, out_filename)
            else:
                continue
            point_cloud = PyntCloud.from_file(file_path)

            # Add GT labels
            points = pd.DataFrame(point_cloud.points[['x','y','z','red','green','blue']])
            points = points.assign(label=-1)
            for (r,g,b), label in COLOR2LABEL.items():
                class_value = LABEL2CLASS[label]
                points.loc[(points['red']==r) & (points['green']==g) & (points['blue']==b), 
                        'label'] = class_value
            points = points[['x','y','z','label']]

            # Transform points from cms to meters
            points['x'] = points['x'].div(100)
            points['y'] = points['y'].div(100)
            points['z'] = points['z'].div(100)

            points = points.to_numpy()
            '''
            print(points.shape)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:,0:3])
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            points = np.asarray(pcd.points)
            print(points.shape)
            exit(1)
            '''
            np.save(room_out_filepath, points)
    
            # Split into blocks
            if folder in config['TRAIN']['HOUSES']:
                blocks_list = room2blocks(points, block_size=config['BLOCK']['SIZE'], 
                                                stride=config['BLOCK']['STRIDE'], 
                                                min_npts=config['BLOCK']['MIN_NPTS'])
                block_out_folderpath = TRAIN_BLOCK_PATH
            elif folder in config['VAL']['HOUSES']:
                blocks_list = room2samples(points, sample_num_point=config['BLOCK']['NPTS'])
                block_out_folderpath = VAL_BLOCK_PATH
            elif folder in config['TEST']['HOUSES']:
                blocks_list = room2samples(points, sample_num_point=config['BLOCK']['NPTS'])
                block_out_folderpath = TEST_BLOCK_PATH
            
            print('{0} is split into {1} blocks.'.format(out_filename, len(blocks_list)))
            for i, block_data in enumerate(blocks_list):
                block_filename = out_filename[:-4] + '_block_' + str(i) + '.npy'
                np.save(os.path.join(block_out_folderpath, block_filename), block_data)

                # === End for loop classes ===
            # === End for loop blocks ===
        # === End for loop rooms ===
    # === End for loop houses ===
    
    # Save pkl mapping file
    with open(PKL_MAP_FILE, 'wb') as f:
        pickle.dump([CLASS_LABELS, LABEL2CLASS], f, pickle.HIGHEST_PROTOCOL)

