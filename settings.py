#!/usr/bin/env python3
"""
Settings


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""
import os

path_all_models = './models'
path_models = './models/pretrained'
path_models_users = './models/users'
path_to_data = './data'

# deam data (change here!)
deam_data = '/media/hoodoochild/DATA/datasets/deam'
# this is only needed to pretrain the models
deam_feats = '{}/features'.format(deam_data)
deam_dataset_fn = '{}/dataset_quads.csv'.format(deam_data)
deam_npy = '{}/npy'.format(deam_data)
deam_anno_arousal = 'deam_annotations/arousal.csv'
deam_anno_valence = 'deam_annotations/valence.csv'


# amg testing (change here!)
amg_data = '/media/hoodoochild/DATA/datasets/amg1608'
path_to_audio_amg = '{}/audio'.format(amg_data)
path_to_feats_amg = '{}/feats'.format(amg_data)
amg_npy = '{}/npy'.format(amg_data)
dataset_fn_amg = '{}/dataset_feats.csv'.format(amg_data)
dataset_anno_amg = '{}/anno/AMG1608.mat'.format(amg_data)
mapping_amg = '{}/anno/1608_song_id.mat'.format(amg_data)

# short cnn configurations
input_length = 59049
n_epochs_cnn = 200
batch_size = 5
lr = 1e-4
log_step = 20
# re-training cnn
n_epochs_retrain = 100

