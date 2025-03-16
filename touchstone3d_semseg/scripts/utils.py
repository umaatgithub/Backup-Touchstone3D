""" Utility functions common to all datasets.

Author: Umamaheswaran Raman Kumar & Abdur R. Fayjie, 2023
"""

import os
import numpy as np
import h5py as h5
import transforms3d
import random
import math


def room2blocks(data, block_size, stride, min_npts):
    """ Prepare block data.
    Args:
    data: N x 7 numpy array, 012 are XYZ in meters, 345 are RGB in [0,255], 6 is the labels
      assumes the data is not shifted (min point is not origin),
    block_size: float, physical size of the block in meters
    stride: float, stride for block sweeping
    Returns:
    blocks_list: a list of blocks, each block is a num_point x 7 np array
    """
    assert (stride <= block_size)

    xyz = data[:,:3]
    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    xyz_max = np.amax(xyz, axis=0)

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((xyz_max[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((xyz_max[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride)
            ybeg_list.append(j * stride)

    # Collect blocks
    blocks_list = []
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (xyz[:, 0] <= xbeg + block_size) & (xyz[:, 0] >= xbeg)
        ycond = (xyz[:, 1] <= ybeg + block_size) & (xyz[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < min_npts:  # discard block if there are less than 100 pts.
            continue

        block = data[cond, :]
        blocks_list.append(block)

    return blocks_list


def room2samples(data, sample_num_point):
    """ Prepare whole room samples.

    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and
            aligned (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        sample_num_point: int, how many points to sample in each sample
    Returns:
        sample_datas: K x sample_num_point x 9
                     numpy array of XYZRGBX'Y'Z', RGB is in [0,1]
        sample_labels: K x sample_num_point x 1 np array of uint8 labels
    """
    N = data.shape[0]
    order = np.arange(N)
    np.random.shuffle(order) 
    data = data[order, :]

    sample_num = int(np.floor(np.ceil(N / float(sample_num_point))))
    
    samples_list = []
    for i in range(sample_num):
        beg_idx = i*sample_num_point
        end_idx = min((i+1)*sample_num_point, N)
        num = end_idx - beg_idx
        sample = data[beg_idx:end_idx, :]
        samples_list.append(sample)    
    return samples_list


def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name):
    data = np.load(os.path.join(data_path, scan_name))
    N = data.shape[0] #number of points in this scan

    sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point))

    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    if 'rgb' in pc_attribs:
        rgb = data[:, 3:6]
    labels = data[:,-1].astype(int)

    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1)

    return ptcloud, labels


def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P


def write_episode(out_filename, data):
    support_ptclouds, support_masks, query_ptclouds, query_labels, sampled_classes = data
    data_file = h5.File(out_filename, 'w')
    data_file.create_dataset('support_ptclouds', data=support_ptclouds, dtype='float32')
    data_file.create_dataset('support_masks', data=support_masks, dtype='int32')
    data_file.create_dataset('query_ptclouds', data=query_ptclouds, dtype='float32')
    data_file.create_dataset('query_labels', data=query_labels, dtype='int64')
    data_file.create_dataset('sampled_classes', data=sampled_classes, dtype='int32')
    data_file.close()

    print('\t {0} saved! | classes: {1}'.format(out_filename, sampled_classes))


def read_episode(file_name):
    data_file = h5.File(file_name, 'r')
    support_ptclouds = data_file['support_ptclouds'][:]
    support_masks = data_file['support_masks'][:]
    query_ptclouds = data_file['query_ptclouds'][:]
    query_labels = data_file['query_labels'][:]
    sampled_classes = data_file['sampled_classes'][:]

    return support_ptclouds, support_masks, query_ptclouds, query_labels, sampled_classes


def batch_train_task_collate(batch):
    task_train_support_ptclouds, task_train_support_masks, task_train_query_ptclouds, task_train_query_labels, \
    task_valid_support_ptclouds, task_valid_support_masks, task_valid_query_ptclouds, task_valid_query_labels = list(zip(*batch))

    task_train_support_ptclouds = np.stack(task_train_support_ptclouds)
    task_train_support_masks = np.stack(task_train_support_masks)
    task_train_query_ptclouds = np.stack(task_train_query_ptclouds)
    task_train_query_labels = np.stack(task_train_query_labels)
    task_valid_support_ptclouds = np.stack(task_valid_support_ptclouds)
    task_valid_support_masks = np.stack(task_valid_support_masks)
    task_valid_query_ptclouds = np.array(task_valid_query_ptclouds)
    task_valid_query_labels = np.stack(task_valid_query_labels)

    data = [torch.from_numpy(task_train_support_ptclouds).transpose(3,4), torch.from_numpy(task_train_support_masks),
            torch.from_numpy(task_train_query_ptclouds).transpose(2,3), torch.from_numpy(task_train_query_labels),
            torch.from_numpy(task_valid_support_ptclouds).transpose(3,4), torch.from_numpy(task_valid_support_masks),
            torch.from_numpy(task_valid_query_ptclouds).transpose(2,3), torch.from_numpy(task_valid_query_labels)]

    return data


def batch_test_task_collate(batch):
    batch_support_ptclouds, batch_support_masks, batch_query_ptclouds, batch_query_labels, batch_sampled_classes = batch[0]

    data = [torch.from_numpy(batch_support_ptclouds).transpose(2,3), torch.from_numpy(batch_support_masks),
            torch.from_numpy(batch_query_ptclouds).transpose(1,2), torch.from_numpy(batch_query_labels.astype(np.int64))]

    return data, batch_sampled_classes
