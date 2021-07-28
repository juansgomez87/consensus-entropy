#!/usr/bin/env python3
"""
Settings


Copyright 2021, J.S. Gómez-Cañón
Licensed under ???
"""
import os

path_models = './models/pretrained'
if 'TPL_INTERNAL_DATA_DIRECTORY' in os.environ:
    path_models_users = os.path.join(os.environ['TPL_INTERNAL_DATA_DIRECTORY'], 'models', 'users')
    path_to_data = os.path.join(os.environ['TPL_INTERNAL_DATA_DIRECTORY'], 'data')
else:
    path_models_users = './models/users'
    path_to_data = './data'

# trompa data
path_to_audio = '{}/audio'.format(path_to_data)
path_to_feats = '{}/feats'.format(path_to_data)
dataset_fn = '{}/dataset_feats.csv'.format(path_to_data)
dataset_anno = '{}/data_04_11_2020.json'.format(path_to_data)

# deam data
# this is only needed to pretrain the models
deam_feats = '/media/hoodoochild/DATA/datasets/deam/features'
deam_dataset_fn = '/media/hoodoochild/DATA/datasets/deam/dataset_quads.csv'
deam_npy = '/media/hoodoochild/DATA/datasets/deam/npy'
deam_anno_arousal = 'deam_annotations/arousal.csv'
deam_anno_valence = 'deam_annotations/valence.csv'

# live experiments
path_models_users_live = './models/live_experiments'
path_models_users_off = './models/off_experiments'

# amg testing
path_to_audio_amg = '/media/hoodoochild/DATA/datasets/amg1608/audio'
path_to_feats_amg = '/media/hoodoochild/DATA/datasets/amg1608/feats'
amg_npy = '/media/hoodoochild/DATA/datasets/amg1608/npy'
dataset_fn_amg = '/media/hoodoochild/DATA/datasets/amg1608/dataset_feats.csv'
dataset_anno_amg = '/media/hoodoochild/DATA/datasets/amg1608/anno/AMG1608.mat'
mapping_amg = '/media/hoodoochild/DATA/datasets/amg1608/anno/1608_song_id.mat'
path_models_amg = './models/amg'

# short cnn configurations
input_length = 59049
n_epochs_cnn = 200
batch_size = 5
lr = 1e-4
log_step = 20
# re-training cnn
n_epochs_retrain = 100

