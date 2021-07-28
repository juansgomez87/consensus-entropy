#!/usr/bin/env python3
"""
Emotion algorithm to test active learning on AMG1608 dataset.

Copyright 2021, J.S. Gómez-Cañón
Licensed under ???
"""

import argparse
import numpy as np
import pandas as pd
import click
import os
import sys
import pdb
import joblib
from collections import Counter
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
import subprocess
import shutil
import gc
import datetime
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import warnings
from scipy.io import loadmat
warnings.filterwarnings("ignore")

from settings import *


class AMG_Tester():
    def __init__(self,
                 epochs,
                 queries,
                 mode,
                 num_anno):
        """Constructor method
        """
        self.path_to_feats = path_to_feats_amg
        self.epochs = np.arange(epochs)
        self.queries = queries
        self.mode = mode
        self.num_anno = num_anno
        self.dict_class = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
        self.seed = np.random.seed(1987)

        if os.path.exists(dataset_fn_amg):
            self.dataset = pd.read_csv(dataset_fn_amg, sep=';')
        else:
            self.dataset = self.load_feats(dataset_fn_amg)
        self.all_ids = self.dataset.s_id.unique().tolist()
        self.all_ids.sort()

        X_pool = StandardScaler().fit_transform(self.dataset.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean'])
        self.X_pool = pd.DataFrame(X_pool, index=self.dataset.s_id)

        self.anno = self.load_annotations(dataset_anno_amg)

    def get_quadrant(self, arousal, valence):
        if arousal >= 0 and valence >= 0:
            quad = 'Q1'
        elif arousal > 0 and valence < 0:
            quad = 'Q2'
        elif arousal <= 0 and valence <= 0:
            quad = 'Q3'
        elif arousal < 0 and valence > 0:
            quad = 'Q4'
        return quad

    def load_models(self, user_path):
        mod_list = [os.path.join(root, f) for root, dirs, files in os.walk(user_path) for f in files if f.lower().endswith('.pkl')]
        if len(mod_list) == 0:
            print('No pre-trained models of this type!')
            sys.exit(0)
        return mod_list

    def load_annotations(self, dataset_anno):
        mat_amg = loadmat(dataset_anno)
        anno = mat_amg['song_label']
        user_ids = np.arange(anno.shape[1])
        mapping = loadmat(mapping_amg)

        gc.collect()
        list_anno = list()
        for s_id, i in zip(mapping['mat_id2song_id'], np.arange(anno.shape[0])):
            this_anno = pd.DataFrame(anno[i], columns=['valence', 'arousal'])
            this_anno['song_id'] = s_id[0]
            this_anno['user_id'] = user_ids
            list_anno.append(this_anno)
        all_anno = pd.concat(list_anno, ignore_index=True)
        all_anno_full = all_anno.dropna().reset_index().drop(columns='index')
        # add quadrants
        aro_list = all_anno_full.arousal.values
        val_list = all_anno_full.valence.values
        quad_list = list(map(self.get_quadrant, aro_list, val_list))
        all_anno_full['quadrant'] = quad_list

        # calculate frequencies for human query-by-commitee + uncertainty sampling
        frequencies = {}
        for id_song in all_anno_full.song_id.unique().tolist():
            cnt_quad = Counter({'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0})
            cnt_quad.update(all_anno_full[all_anno_full.song_id == id_song].quadrant)
            num_anno = all_anno_full.loc[all_anno_full.song_id == id_song].shape[0]
            cnt_quad = dict(cnt_quad)
            frequencies[id_song] = {k: np.round(v / num_anno, 3) for k, v in cnt_quad.items()}
        
        self.consensus_prob_hc = pd.DataFrame(frequencies).transpose()

        count_users_song = {}
        for u_id in user_ids:
            count_users_song[u_id] = all_anno_full[all_anno_full.user_id == u_id].shape[0]
        u_ids_more = [_ for _ in count_users_song.keys() if count_users_song[_]  >= self.num_anno]
        print('Users with more than {} annotations: {}'.format(self.num_anno, len(u_ids_more)))
        cnt_anno = all_anno_full[all_anno_full.user_id.isin(u_ids_more)]
        self.all_users = cnt_anno.user_id.unique().tolist()
        return cnt_anno

    def load_feats(self, dataset_fn):
        fill_char = click.style('=', fg='yellow')
        feats_list = [os.path.join(root, f) for root, dirs, files in os.walk(self.path_to_feats) for f in files if f.lower().endswith('.csv')]
        id_list = [_.split('/')[-1].replace('.csv', '') for _ in feats_list]

        all_feats = []
        with click.progressbar(range(len(id_list)), label='Loading features and processing...', fill_char=fill_char) as bar:
            for feat, s_id, i in zip(feats_list, id_list, bar):
                this_feat = pd.read_csv(feat, sep=';')
                this_feat['s_id'] = s_id
                del this_feat['frameTime']
                all_feats.append(this_feat)

        df_all_feats = pd.concat(all_feats, axis=0)
        df_all_feats.reset_index().drop(columns='index', inplace=True)
        df_all_feats.to_csv(dataset_fn, sep=';', index=False)
        return df_all_feats

    def create_user(self, user, p_mod):
        # create users folder
        path_models_pre = os.path.join(p_mod, 'pretrained')
        path_models_users = os.path.join(p_mod, 'users')
        user_path = os.path.join(path_models_users, str(user), self.mode)
        try:
            os.makedirs(user_path)
        except FileExistsError:
            subprocess.run(['rm', '-rf', user_path])
            os.makedirs(user_path)

        pre_models = [os.path.join(root, f) for root, dirs, files in os.walk(path_models_pre) for f in files if f.lower().endswith('.pkl')]
        cp_models = [f.replace(path_models_pre, user_path) for f in pre_models]

        for in_f, out_f in zip(pre_models, cp_models):
            shutil.copy(in_f, out_f)
        return user_path

    def run(self):
        for num_user, u_id in enumerate(self.all_users):
            # create user folder
            user_path = self.create_user(u_id, path_models_amg)
            # load models
            mod_list = self.load_models(user_path)
            # get songs annotated by this user to reduce batch
            this_anno_user = self.anno[self.anno.user_id == u_id]
            this_song_ids = this_anno_user.song_id.unique().tolist()
            this_X = self.X_pool[self.X_pool.index.isin(this_song_ids)].sort_index()
            this_y = this_anno_user[['song_id', 'quadrant']].set_index('song_id').reindex(this_X.index)
            # reduce human consensus matrix to only the songs annotated by the user
            this_consensus_hc = self.consensus_prob_hc[self.consensus_prob_hc.index.isin(this_song_ids)]
            print('Creating and performing active learning for user {} with {} annotations.'.format(u_id, this_anno_user.shape[0]))
            print('User {} / {}'.format(num_user, len(self.all_users) - 1))
            # split into train and test
            gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=self.seed)
            train_idx, test_idx = next(gss.split(this_X, this_y, this_y.index))
            X_train, y_train = this_X.iloc[train_idx], this_y.iloc[train_idx]
            X_test, y_test = this_X.iloc[test_idx], this_y.iloc[test_idx]
            y_test_enc = y_test.quadrant.replace(self.dict_class)
            # store results in text files
            day_in_time = datetime.datetime.now().strftime("%d-%m-%Y.%H:%M:%S")
            txt_fn = '{}.trial.date_{}.txt'.format(self.mode, day_in_time)
            txt_file = open(os.path.join(user_path, txt_fn), 'a')

            # start re training with epochs
            fill_char = click.style('=', fg='yellow')
            with click.progressbar(range(len(self.epochs)), label='Retraining models...', fill_char=fill_char) as bar:
                for epoch, ba in zip(self.epochs, bar):


                    if epoch == 0:
                        f1_list = list()
                        txt_file.write('---------------------------------')
                        txt_file.write('\n\n~~~~~~~~~\nEpoch {}:~~~~~~~~~\n~~~~~~~~~\n\n\n'.format(epoch - 1))
                        #initial evaluation
                        # re train models and evaluate
                        for mod_fn in mod_list:
                            mod = joblib.load(mod_fn)
                            # test using testing data
                            txt_file.write('Model: {}\n'.format(mod_fn))
                            this_mod_pred = mod.predict(X_test.values)
                            cl_re = classification_report(y_test_enc.values, this_mod_pred)
                            txt_file.write('{}\n'.format(cl_re))
                            f1_list.append(f1_score(y_test_enc.values, this_mod_pred, average='weighted'))

                        txt_file.write('**\nSummary: F1 mean score over all classifiers = {}\n**\n'.format(np.mean(f1_list)))

                    txt_file.write('---------------------------------')
                    txt_file.write('\n\n~~~~~~~~~\nEpoch {}:~~~~~~~~~\n~~~~~~~~~\n\n\n'.format(epoch))

                    print('\nEpoch {}/{}'.format(epoch, len(self.epochs) - 1))

                    if self.mode == 'mc':
                        # machine disagreement-based consensus entropy
                        pred_prob = []
                        for i, mod_fn in enumerate(mod_list):
                            # print('Predicting probabilities for model {} ({}/{})'.format(mod_fn, i, len(self.mod_list)))
                            mod = joblib.load(mod_fn)
                            y_probs = mod.predict_proba(X_train.values)
                            # summarize with mean across all samples
                            y_probs = pd.DataFrame(y_probs, index=X_train.index).groupby(['s_id']).mean()
                            pred_prob.append(y_probs)
                            gc.collect()
                    
                        consensus_prob = np.mean(np.array(pred_prob), axis=0)
                        # entropy calculation
                        ent = entropy(consensus_prob, axis=1)
                        # select songs with max entropy for self.queries amount
                        q_ind = np.argsort(ent)[::-1][:self.queries]
                        # select songs from the average of output probabilities
                        q_songs = y_probs.iloc[q_ind].index.tolist()

                    elif self.mode == 'hc':
                        # select query songs according to human consensus
                        ent_hc = entropy(this_consensus_hc, axis=1)
                        q_ind = np.argsort(ent_hc)[::-1][:self.queries]
                        q_songs = this_consensus_hc.iloc[q_ind].index.tolist()
                        # remove songs from this batch
                        this_consensus_hc = this_consensus_hc[~this_consensus_hc.index.isin(q_songs)]

                    elif self.mode == 'mix':
                        # machine disagreement-based consensus entropy
                        pred_prob = []
                        for i, mod_fn in enumerate(mod_list):
                            # print('Predicting probabilities for model {} ({}/{})'.format(mod_fn, i, len(self.mod_list)))
                            mod = joblib.load(mod_fn)
                            y_probs = mod.predict_proba(X_train.values)
                            # summarize with mean across all samples
                            y_probs = pd.DataFrame(y_probs, index=X_train.index).groupby(['s_id']).mean()
                            pred_prob.append(y_probs)
                            gc.collect()
                    
                        consensus_prob_mc = pd.DataFrame(np.mean(np.array(pred_prob), axis=0), 
                                                         columns=['Q1', 'Q2', 'Q3', 'Q4'],
                                                         index=y_probs.index)
                        
                        mix_consensus = pd.concat([consensus_prob_mc, this_consensus_hc])
                        # entropy calculation
                        ent_mix = entropy(mix_consensus, axis=1)
                        q_ind = np.argsort(ent_mix)[::-1][:self.queries]
                        q_songs = mix_consensus.iloc[q_ind].index.tolist()

                        # remove songs from hc consensus and mc_consensus
                        this_consensus_hc = this_consensus_hc[~this_consensus_hc.index.isin(q_songs)]

                    elif self.mode == 'rand':
                        pos_songs = X_train.index.unique().tolist()
                        np.random.shuffle(pos_songs)
                        q_songs = pos_songs[:self.queries]

                    X_batch = X_train[X_train.index.isin(q_songs)]
                    y_batch = y_train[y_train.index.isin(q_songs)]
                    y_batch_enc = y_batch.quadrant.replace(self.dict_class)
                    f1_list = list()
                    # re train models and evaluate
                    for mod_fn in mod_list:

                        mod = joblib.load(mod_fn)
                        mod_type = mod_fn.split('/')[-1].split('.')[0]
                        if mod_type == 'classifier_xgb':
                            mod.fit(X_batch.values, np.squeeze(y_batch_enc.values), xgb_model=mod.get_booster())
                        elif mod_type == 'classifier_gnb' or mod_type == 'classifier_sgd':
                            mod.partial_fit(X_batch.values, np.squeeze(y_batch_enc.values))
                        # save model
                        joblib.dump(mod, mod_fn)
                        # test using testing data
                        txt_file.write('Model: {}\n'.format(mod_fn))
                        this_mod_pred = mod.predict(X_test.values)
                        # pdb.set_trace()
                        cl_re = classification_report(y_test_enc.values, this_mod_pred)
                        txt_file.write('{}\n'.format(cl_re))
                        f1_list.append(f1_score(y_test_enc.values, this_mod_pred, average='weighted'))

                    txt_file.write('**\nSummary: F1 mean score over all classifiers = {}\n**\n'.format(np.mean(f1_list)))
                    print(q_songs, X_batch.shape[0])
                    # remove batch from pool
                    X_train = X_train.drop(X_batch.index)
                    gc.collect()
                    txt_file.write('---------------------------------')

            txt_file.close()
            gc.collect()
            # pdb.set_trace()


if __name__ == "__main__":
    # Usage
    # full use: python3 amg_test.py -q 15 -e 10 -m mc
    # all exp: python3 amg_test.py -q 15 -e 10 -m rand && python3 amg_test.py -q 15 -e 10 -m mc && python3 amg_test.py -q 15 -e 10 -m hc && python3 amg_test.py -q 15 -e 10 -m mix
    parser = argparse.ArgumentParser()

    parser.add_argument('-q',
                        '--queries',
                        help='Select number of queries to perform (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='queries')
    parser.add_argument('-e',
                        '--epochs',
                        help='Select number of epochs to perform (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='epochs')
    parser.add_argument('-n',
                        '--num_anno',
                        help='Select minimum number of annotations per user (int)',
                        action='store',
                        required=True,
                        type=int,
                        dest='num_anno')
    parser.add_argument('-m',
                        '--mode',
                        help='Select mode of function: machine-consensus [mc], human consensus [hc], both [mix], or random [rand]',
                        action='store',
                        required=True,
                        dest='mode')

    args = parser.parse_args()

    if args.mode != 'hc' and args.mode != 'mc' and args.mode != 'mix' and args.mode != 'rand':
        print('Select a valid consensus calculation mode!')
        sys.exit()

    args = parser.parse_args()

    amg_test = AMG_Tester(args.epochs, args.queries, args.mode, args.num_anno)

    amg_test.run()

