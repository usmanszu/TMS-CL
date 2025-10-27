# """
# This code is adapted/modified from the work:
# Xie Yulai's UAVHuman_For_TE-GCN project (https://github.com/xieyulai/UAVHuman_For_TE-GCN)
# """
#
# """
# Script to process raw data and generate dataset's binary files:
#     - .npy skeleton data files: np.array of shape B x C x V x T x M
#     - .pkl label files: (filename: str, label: list[int])
# """
#
# import argparse
# import pickle
# import os
# import glob
# import re
# import numpy as np
# from tqdm import tqdm
# from preprocess import pre_normalization
#
# # ===========================
# # Configurations
# # ===========================
# MAX_BODY_TRUE = 2
# MAX_BODY_KINECT = 4
# NUM_JOINT = 17
# MAX_FRAME = 300
#
# # Filename pattern to extract action ID (Axx)
# FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'
#
#
# # ===========================
# # Skeleton Reading Functions
# # ===========================
# def read_skeleton_filter(file):
#     with open(file, 'r') as f:
#         skeleton_sequence = {}
#         skeleton_sequence['numFrame'] = int(f.readline())
#         skeleton_sequence['frameInfo'] = []
#         for t in range(skeleton_sequence['numFrame']):
#             frame_info = {}
#             frame_info['numBody'] = int(f.readline())
#             frame_info['bodyInfo'] = []
#
#             for m in range(frame_info['numBody']):
#                 body_info_key = [
#                     'bodyID', 'clipedEdges', 'handLeftConfidence',
#                     'handLeftState', 'handRightConfidence', 'handRightState',
#                     'isResticted', 'leanX', 'leanY', 'trackingState'
#                 ]
#                 body_info = {
#                     k: float(v)
#                     for k, v in zip(body_info_key, f.readline().split())
#                 }
#                 body_info['numJoint'] = int(f.readline())
#                 body_info['jointInfo'] = []
#
#                 for v in range(body_info['numJoint']):
#                     joint_info_key = [
#                         'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
#                         'orientationW', 'orientationX', 'orientationY',
#                         'orientationZ', 'trackingState'
#                     ]
#                     joint_info = {
#                         k: float(v)
#                         for k, v in zip(joint_info_key, f.readline().split())
#                     }
#                     body_info['jointInfo'].append(joint_info)
#                 frame_info['bodyInfo'].append(body_info)
#
#             skeleton_sequence['frameInfo'].append(frame_info)
#     return skeleton_sequence
#
#
# def get_nonzero_std(s):
#     index = s.sum(-1).sum(-1) != 0  # select valid frames
#     s = s[index]
#     if len(s) != 0:
#         s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()
#     else:
#         s = 0
#     return s
#
#
# def read_xyz(file, max_body, num_joint):
#     seq_info = read_skeleton_filter(file)
#     data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
#     for n, f in enumerate(seq_info['frameInfo']):
#         for m, b in enumerate(f['bodyInfo']):
#             for j, v in enumerate(b['jointInfo']):
#                 if m < max_body and j < num_joint:
#                     data[m, n, j, :] = [v['x'], v['y'], v['z']]
#     # Select two most energetic bodies
#     energy = np.array([get_nonzero_std(x) for x in data])
#     index = energy.argsort()[::-1][0:MAX_BODY_TRUE]
#     data = data[index]
#     data = data.transpose(3, 1, 2, 0)
#     return data
#
#
# # ===========================
# # Main Dataset Generator
# # ===========================
# def gendata(data_path, split):
#     out_path = data_path
#     data_path = os.path.join(data_path, split)
#
#     skeleton_filenames = [os.path.basename(f) for f in
#                           glob.glob(os.path.join(data_path, "**.txt"), recursive=True)]
#
#     sample_name = []
#     for basename in skeleton_filenames:
#         filename = os.path.join(data_path, basename)
#         if not os.path.exists(filename):
#             raise OSError('%s does not exist!' % filename)
#         sample_name.append(filename)
#
#     num_of_frames = []
#     data = np.zeros((len(sample_name), 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE), dtype=np.float32)
#
#     for i, s in enumerate(tqdm(sample_name)):
#         sample = read_xyz(s, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT)
#         num_of_frames.append(min(sample.shape[1], MAX_FRAME))
#         sample = sample[:, :MAX_FRAME, :, :]
#         data[i, :, 0:sample.shape[1], :, :] = sample
#
#     data = pre_normalization(data)
#
#     np.save(f'{out_path}/{split}_data.npy', data)
#     np.save(f'{out_path}/{split}_num_of_frames.npy', np.array(num_of_frames, dtype=np.int32))
#
#     sample_label = []
#     for basename in skeleton_filenames:
#         label = int(re.match(FILENAME_REGEX, basename).groups()[0])
#         sample_label.append(label)
#
#     with open(f'{out_path}/{split}_label.pkl', 'wb') as f:
#         pickle.dump((sample_name, list(sample_label)), f)
#
#     print(f"\n✅ Finished generating {split} data:")
#     print(f"  Data shape: {data.shape}")
#     print(f"  Labels: {len(sample_label)} samples")
#
#
# # ===========================
# # Entry Point
# # ===========================
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='UAVHuman Data Converter.')
#     parser.add_argument('--data_path', default='../data/UAVHuman/ActionRecognition/Skeleton',
#                         help='Path to UAVHuman skeleton dataset')
#     args = parser.parse_args()
#
#     gendata(data_path=args.data_path, split='train')
#     gendata(data_path=args.data_path, split='test')

# to create
"""
This code is adapted/modified from the work:
Xie Yulai's UAVHuman_For_TE-GCN project (https://github.com/xieyulai/UAVHuman_For_TE-GCN)
"""

"""
Script to process raw data and generate dataset's binary files:
    - .npy skeleton data files: np.array of shape B x C x T x V x M
    - .pkl label files: (filename: str, label: list[int])
"""

import argparse
import pickle
import os
import glob
import re
import numpy as np
from tqdm import tqdm
from preprocess import pre_normalization

# ===========================
# Configurations
# ===========================
MAX_BODY_TRUE   = 2
MAX_BODY_KINECT = 4
NUM_JOINT       = 17
MAX_FRAME       = 300

# Filename pattern to extract action ID (Axx)
# e.g., P001S01G01B01H01UC01LC01A12R01_0001.txt  -> label = 12
FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'

# ===========================
# Skeleton Reading Functions
# ===========================
def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        for _ in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for _m in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {k: float(v) for k, v in zip(body_info_key, f.readline().split())}
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                joint_info_key = [
                    'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                    'orientationW', 'orientationX', 'orientationY',
                    'orientationZ', 'trackingState'
                ]
                for _v in range(body_info['numJoint']):
                    joint_info = {k: float(v) for k, v in zip(joint_info_key, f.readline().split())}
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def get_nonzero_std(s):
    # s: (T, V, 3); select frames with any non-zero joint
    idx = s.sum(-1).sum(-1) != 0
    s = s[idx]
    return (s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()) if len(s) != 0 else 0.0

def read_xyz(file, max_body, num_joint):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3), dtype=np.float32)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
    # keep top-2 energetic bodies
    energy = np.array([get_nonzero_std(x) for x in data])
    keep = energy.argsort()[::-1][:MAX_BODY_TRUE]
    data = data[keep]
    return data.transpose(3, 1, 2, 0)  # (C=3, T, V, M)

# ===========================
# IO helpers
# ===========================
def _glob_txts(root):
    # recursive, case-insensitive
    pats = [os.path.join(root, '**', '*.txt'), os.path.join(root, '**', '*.TXT')]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))

def _ensure_split_dirs(data_path):
    train_dir = os.path.join(data_path, 'train')
    test_dir  = os.path.join(data_path, 'test')
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"[ERROR] data_path does not exist: {data_path}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"[ERROR] split dir not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"[ERROR] split dir not found: {test_dir}")
    return train_dir, test_dir

# ===========================
# Main Dataset Generator
# ===========================
def gendata(data_path, split):
    # Validate folders
    train_dir, test_dir = _ensure_split_dirs(data_path)
    split_dir = train_dir if split == 'train' else test_dir
    out_path  = data_path  # write binaries alongside the two split folders

    files = _glob_txts(split_dir)
    print(f"[INFO] {split}: found {len(files)} skeleton files under {split_dir}")
    if len(files) == 0:
        raise RuntimeError(f"[ERROR] No .txt files found in {split_dir}. Check your split folders.")

    N = len(files)
    data = np.zeros((N, 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE), dtype=np.float32)
    num_of_frames, sample_name, labels = [], [], []

    for i, fpath in enumerate(tqdm(files, desc=f"Building {split}")):
        sample = read_xyz(fpath, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT)  # (3,T,V,M)
        T = min(sample.shape[1], MAX_FRAME)
        data[i, :, :T, :, :] = sample[:, :T, :, :]
        num_of_frames.append(T)
        sample_name.append(fpath)  # store full path for traceability

        base = os.path.basename(fpath)
        m = re.match(FILENAME_REGEX, base)
        if not m:
            raise ValueError(
                f"[LABEL PARSE ERROR] Filename does not match FILENAME_REGEX:\n"
                f"  file : {base}\n  regex: {FILENAME_REGEX}\n"
                f"→ Rename files to include 'A<label>' or modify FILENAME_REGEX."
            )
        labels.append(int(m.groups()[0]))

    # normalization on (N, C, T, V, M)
    data = pre_normalization(data)

    # save
    np.save(os.path.join(out_path, f'{split}_data.npy'), data)
    np.save(os.path.join(out_path, f'{split}_num_of_frames.npy'), np.array(num_of_frames, dtype=np.int32))
    with open(os.path.join(out_path, f'{split}_label.pkl'), 'wb') as f:
        pickle.dump((sample_name, labels), f)

    print(f"\n✅ Finished generating {split} data:")
    print(f"  Data shape : {data.shape}")
    print(f"  Num frames : min={min(num_of_frames)} max={max(num_of_frames)}")
    print(f"  Labels     : {len(labels)} samples")
    print(f"  Output dir : {out_path}")

# ===========================
# Entry Point
# ===========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UAVHuman Data Converter.')
    parser.add_argument(
        '--data_path',
        default='../data/UAVHuman/benchmarks/uav-cross-subjectv1',
        help='Path containing split folders: <data_path>/train and <data_path>/test'
    )
    args = parser.parse_args()

    # Example: for v2, run again with --data_path ../data/UAVHuman/benchmarks/uav-cross-subjectv2
    gendata(data_path=args.data_path, split='train')
    gendata(data_path=args.data_path, split='test')
