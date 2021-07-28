#!/usr/bin/env python3
"""
Emotion algorithm pre-trainer using DEAM data


Copyright 2021, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""

import argparse
import numpy as np
import pandas as pd
# import modin.pandas as pd
import re
import os
import click
import pdb
import joblib

from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

import torch
from torch.utils import tensorboard
import time
import datetime

from settings import *

class TROMPAClassifier():
    def __init__(self,
                 cross_val,
                 model):
        """Constructor method
        """
        self.cv = cross_val
        self.path_to_feats = deam_feats
        self.anno_arousal = deam_anno_arousal
        self.anno_valence = deam_anno_valence
        self.seed = np.random.seed(1987)
        self.model = model
        if os.path.exists(deam_dataset_fn):
            self.dataset = pd.read_csv(deam_dataset_fn)
        else:
            self.dataset = self.load_dataset()


    def load_dataset(self):
        fill_char = click.style('=', fg='yellow')
        feats_list = [os.path.join(root, f) for root, dirs, files in os.walk(self.path_to_feats) for f in files if f.lower().endswith('.csv')]
        feats_list.sort(key=lambda f: int(re.sub('\D', '', f)))
        id_list = [int(_.split('/')[-1].replace('.csv', '')) for _ in feats_list]

        arousal = pd.read_csv(self.anno_arousal, sep=',')
        valence = pd.read_csv(self.anno_valence, sep=',')

        dataset = []
        with click.progressbar(range(len(id_list)), label='Assembling dataset...', fill_char=fill_char) as bar:
            for feat, s_id, i in zip(feats_list, id_list, bar):
                this_feat = pd.read_csv(feat, sep=';')
                this_aro = arousal[arousal.song_id == s_id].dropna(axis=1)
                samp_aro = [int(_.replace('sample_', '').replace('00ms', ''))/10 for _ in this_aro.columns.tolist()[1:]]
                this_val = valence[valence.song_id == s_id].dropna(axis=1)
                samp_val= [int(_.replace('sample_', '').replace('00ms', ''))/10 for _ in this_val.columns.tolist()[1:]]
                if samp_aro != samp_val:
                    # print('Something is wrong with length of the annotations!')
                    if len(samp_aro) > len(samp_val):
                        samp_all = samp_val
                    else:
                        samp_all = samp_aro
                    # pdb.set_trace()
                else:
                    samp_all = samp_aro
                sliced_feat = this_feat[this_feat.frameTime.isin(samp_all)].copy()
                cols = ['sample_{}00ms'.format(int(_*10)) for _ in sliced_feat.frameTime.values.tolist()]
                sliced_feat['arousal'] = this_aro.loc[:, cols].values[0]
                sliced_feat['valence'] = this_val.loc[:, cols].values[0]
                quads = []
                for idx, row in sliced_feat.iterrows():
                    if row['arousal'] >= 0 and row['valence'] >= 0:
                        quads.append('Q1')
                    elif row['arousal'] >= 0 and row['valence'] < 0:
                        quads.append('Q2')
                    elif row['arousal'] < 0 and row['valence'] < 0:
                        quads.append('Q3')
                    elif row['arousal'] < 0 and row['valence'] >= 0:
                        quads.append('Q4')
                sliced_feat['quadrants'] = quads
                sliced_feat['song_id'] = s_id
                dataset.append(sliced_feat)
        df = pd.concat(dataset)
        df.reset_index().drop(columns='index', inplace=True)
        df.to_csv(deam_dataset_fn, index=False)
        return df

    def validation(self, model, epoch, loss, loader, best_metric, mod_fn):
        model = model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = loss

        for x, y in loader:
            # Forward
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)

            out = model(x)

            # Backward
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(np.mean(y.detach().cpu().numpy(), axis=0))

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss_mean = np.mean(losses)
        print('loss: %.4f' % loss_mean)

        f1_sc = f1_score(np.argmax(gt_array, axis=1), np.argmax(est_array, axis=1), average='weighted')
        print('f1_score: %.4f' % f1_sc)

        score = 1 - loss_mean
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(model.state_dict(), mod_fn)
        return f1_sc, loss_mean, best_metric


    def opt_schedule(self, model, mod_fn, current_optimizer, drop_counter, opt):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 40:
            state = torch.load(mod_fn)
            model.load_state_dict(state)
            opt = torch.optim.SGD(model.parameters(), 0.001,
                                  momentum=0.9, weight_decay=0.0001,
                                  nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            state = torch.load(mod_fn)
            model.load_state_dict(state)
            for pg in opt.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            state = torch.load(mod_fn)
            model.load_state_dict(state)
            for pg in opt.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter, opt


    def run(self):
        # pre-extracted feature sets (i.e. deam dataset)
        if 'pcm_fftMag_mfcc_sma_de[14]_amean' in self.dataset.columns.tolist():
            X = self.dataset.loc[:, 'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
        # features extracted with this library
        elif 'mfcc_sma_de[14]_amean' in self.dataset.columns.tolist():
            X = self.feats.loc[:, 'F0final_sma_stddev':'mfcc_sma_de[14]_amean']
        else:
            print('Something is wrong with the input features. Exiting!')
            sys.exit(0)

        y = self.dataset.loc[:, 'quadrants']
        song_ids = self.dataset.loc[:, 'song_id']

        feat_scale = True
        if feat_scale:
            X.loc[:, :] = StandardScaler().fit_transform(X)

        y_enc = LabelEncoder().fit_transform(y)

        kfold = GroupShuffleSplit(n_splits=self.cv, random_state=self.seed)

        if self.model == 'rf':
            model = RandomForestClassifier(random_state=self.seed, warm_start=True)
            txt_fn = 'classifier_rf'
        elif self.model == 'svc':
            model = SVC(probability=True, random_state=self.seed)
            txt_fn = 'classifier_svm'
        elif self.model == 'knn':
            model = KNeighborsClassifier()
            txt_fn = 'classifier_knn'
        elif self.model == 'gnb':
            model = GaussianNB()
            txt_fn = 'classifier_gnb'
        elif self.model == 'sgd':
            model = SGDClassifier(loss='log',
                                  penalty='l2', 
                                  random_state=self.seed, 
                                  warm_start=True)
            txt_fn = 'classifier_sgd'
        elif self.model == 'gpc':
            kernel = 1.0 * RBF(1.0)
            model = GaussianProcessClassifier(kernel=kernel, random_state=self.seed, warm_start=True)
            txt_fn = 'classifier_gpc'
        elif self.model == 'gbc':
            model = GradientBoostingClassifier(max_depth=2, random_state=self.seed, warm_start=True)
            txt_fn = 'classifier_gbc'
        elif self.model == 'xgb':
            model = XGBClassifier(random_state=self.seed,
                                  max_depth=5,
                                  use_label_encoder=False,
                                  eval_metric='auc',
                                  nthread=4)
            txt_fn = 'classifier_xgb'
            X = X.to_numpy()
        elif self.model == 'cnn':
            print('Since model is too heavy, no cross-validation will be performed!')

            from short_cnn import ShortChunkCNN, get_audio_loader

            model = ShortChunkCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            current_optimizer = 'adam'
            writer = tensorboard.SummaryWriter()

            if torch.cuda.is_available():
                model.cuda()
            txt_fn = 'classifier_cnn'

        # training and validation
        if self.model == 'cnn':
            # tr, te = next(kfold.split(X, y, song_ids))
            for it, (tr, te) in enumerate(kfold.split(X, y, song_ids)):
                filename = 'models/pretrained/{}.it_{}.pth'.format(txt_fn, it)
                id_tr = self.dataset.loc[:, 'quadrants':'song_id'].iloc[tr].groupby(['song_id']).max()
                id_te = self.dataset.loc[:, 'quadrants':'song_id'].iloc[te].groupby(['song_id']).max()

                train_loader = get_audio_loader(deam_npy,
                                                batch_size,
                                                split=id_tr,
                                                input_length=input_length,
                                                num_workers=1)
                test_loader = get_audio_loader(deam_npy,
                                               batch_size=1,
                                               split=id_te,
                                               input_length=input_length,
                                               num_workers=1)

                # start training
                start_t = time.time()
                loss_fun = torch.nn.BCELoss()
                best_metric = 0
                drop_counter = 0

                for epoch in range(n_epochs_cnn):
                    ctr = 0
                    drop_counter += 1
                    model = model.train()
                    for x, y in train_loader:
                        ctr += 1
                        # Forward
                        if torch.cuda.is_available():
                            x = x.cuda()
                            y = y.cuda()
                        x = torch.autograd.Variable(x)
                        y = torch.autograd.Variable(y)

                        out = model(x)

                        # Backward
                        loss = loss_fun(out, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if (ctr) % log_step == 0:
                            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        epoch+1, n_epochs_cnn, ctr, len(train_loader), loss.item(),
                                        datetime.timedelta(seconds=time.time()-start_t)))

                    # validation
                    f1_sc, loss_mean, best_metric = self.validation(model,
                                                                    epoch,
                                                                    loss_fun,
                                                                    test_loader,
                                                                    best_metric,
                                                                    filename)

                    # schedule optimizer
                    current_optimizer, drop_counter, optimizer = self.opt_schedule(model,
                                                                                   filename,
                                                                                   current_optimizer,
                                                                                   drop_counter,
                                                                                   optimizer)

                    writer.add_scalar('Loss/train', loss.item(), epoch)
                    writer.add_scalar('Loss/valid', loss_mean, epoch)
                    writer.add_scalar('AUC/f1', f1_sc, epoch)

        else:
            scoring = ['precision_weighted', 'recall_weighted', 'f1_weighted']
            res = cross_validate(model,
                                 X,
                                 y_enc,
                                 cv=kfold,
                                 groups=song_ids,
                                 scoring=scoring,
                                 n_jobs=10,
                                 return_estimator=True,
                                 verbose=10,)

            # TODO: only save best model??
            for i, model in enumerate(res['estimator']):
                filename = 'models/pretrained/{}.it_{}.pkl'.format(txt_fn, i)
                joblib.dump(model, filename)

            print('\n*-*-*-*-*-*-*-\n*-*-*-*-*-*-*-\n CV RESULTS\n*-*-*-*-*-*-*-\n*-*-*-*-*-*-*-')
            print('PRECISION: {0:.3f} ± {1:.3f} ({2:.3f})'.format(res['test_precision_weighted'].mean(),
                                                                  2 * res['test_precision_weighted'].std(),
                                                                  res['test_precision_weighted'].std()))
            print('RECALL: {0:.3f} ± {1:.3f} ({2:.3f})'.format(res['test_recall_weighted'].mean(),
                                                                  2 * res['test_recall_weighted'].std(),
                                                                  res['test_recall_weighted'].std()))
            print('F1 SCORE: {0:.3f} ± {1:.3f} ({2:.3f})'.format(res['test_f1_weighted'].mean(),
                                                                  2 * res['test_f1_weighted'].std(),
                                                                  res['test_f1_weighted'].std()))

            y_pred = res['estimator'][0].predict(X)
            y_prob = model.predict_proba(X)
            print(classification_report(y_enc, y_pred))

        pdb.set_trace()


if __name__ == "__main__":
    # usage: python3 deam_classifier.py -cv CVAL_SPLIT -m MODEL_TYPE
    # example: python3 deam_classifier.py -cv 5 -m xgb
    parser = argparse.ArgumentParser()
    parser.add_argument('-cv',
                        '--cross_val',
                        help='Select cross validation split (int)',
                        action='store',
                        required=True,
                        dest='cross_val')
    parser.add_argument('-m',
                        '--model',
                        help='Select model to train: K nearest neighbors [knn], Gaussian Naive Bayes [gnb], Gaussian Process RBF [gpc], Support Vector Machine [svc], Random Forest [rf], Gradient Boosting [gbc], Stochastic Gradient Descent [sgd], xgboost [xgb], short-chunk cnn [cnn]',
                        action='store',
                        required=True,
                        dest='model')
    args = parser.parse_args()


    try:
        cross_val = int(args.cross_val)
    except ValueError:
        print('Cross validation parameter must be a number!')
        sys.exit()

    if args.model != 'knn' and args.model != 'gnb' and args.model != 'gpc' and args.model != 'svc' and args.model != 'rf' and args.model != 'gbc' and args.model != 'sgd' and args.model != 'xgb' and args.model != 'cnn':
        print('Select a valid model!')
        sys.exit()

    clf = TROMPAClassifier(cross_val, args.model)

    clf.run()
