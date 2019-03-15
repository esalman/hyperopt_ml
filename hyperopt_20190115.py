# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:02:37 2018

@author: msalman
"""

import datetime
from hyperopt import fmin, tpe, hp, anneal, Trials
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
import pickle
from scipy.io import loadmat
from scipy.stats import ttest_1samp, ttest_ind
import seaborn as sns
from sklearn import svm, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.feature_selection import RFE, RFECV, f_classif, SelectPercentile, SelectKBest
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, LeaveOneOut
#from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from statsmodels.stats.multitest import multipletests
import sys
import time
import xgboost as xgb

start_time= time.time()
rcParams['figure.dpi'] = 75

random_state = None

num_comp = 47
num_sub = 314
num_voxel = 60303
repeats = 1
ttest1_intersect = 1
ttest2_intersect = 1
metric_ = 'accuracy'
metrics_ = ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'AUC']
test_size = .3
valid_size = .3
n_iters = 50
cv_k = 5

ttest1_corr_method = 'b'
ttest1_corr_alpha = .001
ttest2_corr_method = 'None'
ttest2_corr_alpha = .01

cfg_load_maps               = 0
cfg_select_feats_ttest1     = 0
cfg_plot_all_strategy       = 0
cfg_quick                   = 1

network_path = '../results/zfu_data_analysis/'
outpath = '../results/zfu_data_analysis/classification_py/'
fig_ext = '.png'

if not os.path.exists(outpath):
    os.makedirs(outpath)

# load data and labels
meta_ = '../results/meta_314_subjects.csv'
y = pd.read_csv(meta_, header=None)
y = y.iloc[:,3]
y[y==2] = -1

def t():
    h = '%02d'%datetime.datetime.now().hour
    m = '%02d'%datetime.datetime.now().minute
    s = '%02d'%datetime.datetime.now().second
    return h+':'+m+':'+s

class NetworkFeaturesTtest(BaseEstimator, TransformerMixin):
    def __init__(self, X2):
        self.X2 = X2
        
    def fit(self, X, y):
        X2 = np.reshape( np.array(self.X2.loc[X.index,:]), (-1, num_comp, num_voxel) )
        X = np.reshape( np.array(X), (-1, num_comp, num_voxel) )
        feat_sel = []
        
        for i in range(num_comp):
            # ttest1
            _, prob = ttest_1samp(X[:,i,:], 0)
            rejects, prob_adj, _, _ = multipletests(prob, method=ttest1_corr_method, alpha=ttest1_corr_alpha)
            ttest1_signf_idx_X1 = np.where(rejects)[0]
            
            _, prob = ttest_1samp(X2[:,i,:], 0)
            rejects, prob_adj, _, _ = multipletests(prob, method=ttest1_corr_method, alpha=ttest1_corr_alpha)
            ttest1_signf_idx_X2 = np.where(rejects)[0]
            
            if ttest1_intersect:
                ttest1_signf_idx_X1 = np.intersect1d(ttest1_signf_idx_X1, ttest1_signf_idx_X2)
                ttest1_signf_idx_X2 = ttest1_signf_idx_X1
                
            # ttest2
            t1 = np.squeeze(X[:,i,ttest1_signf_idx_X1])
            _, prob = ttest_ind(t1[y==1,:], t1[y==-1,:])
            if ttest2_corr_method == 'None':
                ttest2_signf_idx_X1 = ttest1_signf_idx_X1[np.where(prob < ttest2_corr_alpha)[0]]
            else:
                rejects, prob_adj, _, _ = multipletests(prob, method=ttest2_corr_method, alpha=ttest2_corr_alpha)
                ttest2_signf_idx_X1 = ttest1_signf_idx_X1[np.where(rejects)[0]]
            
            t1 = np.squeeze(X2[:,i,ttest1_signf_idx_X2])
            _, prob = ttest_ind(t1[y==1,:], t1[y==-1,:])
            if ttest2_corr_method == 'None':
                ttest2_signf_idx_X2 = ttest1_signf_idx_X2[np.where(prob < ttest2_corr_alpha)[0]]
            else:
                rejects, prob_adj, _, _ = multipletests(prob, method=ttest2_corr_method, alpha=ttest2_corr_alpha)
                ttest2_signf_idx_X2 = ttest1_signf_idx_X2[np.where(rejects)[0]]
            
#            # RFE
#            estimator = svm.SVC(kernel='linear')
#            selector = RFE(estimator=estimator, step=.05, n_features_to_select=20)
#            t1 = np.squeeze(X[:,i,ttest1_signf_idx_X1])
#            selector.fit(t1, y)
#            ttest2_signf_idx_X1 = ttest1_signf_idx_X1[np.where( selector.support_ )]
            
            if ttest2_intersect:
                ttest2_signf_idx_X1 = np.intersect1d(ttest2_signf_idx_X1, ttest2_signf_idx_X2)
            
            feat_sel.append( (i*num_voxel)+ttest2_signf_idx_X1 )
        
        self.support_ = np.hstack(feat_sel)
        print('selected features:', self.support_.size)
        return self
    
    def transform(self, X):
        return X.loc[:, self.support_]

# possible values of parameters
#space = {
#    'min_child_weight': [1, 5, 10],
#    'gamma': [0.5, 1, 1.5, 2, 5],
#    'subsample': [0.6, 0.8, 1.0],
#    'colsample_bytree': [0.6, 0.8, 1.0],
#    'max_depth': [3, 4, 5]
#    }

def classify_(X_train, X_test, y_train, y_test):
    def gb_mse_cv(params, random_state=None, cv=KFold(n_splits=cv_k, random_state=None), 
                  X=X_train, y=y_train):
        # the function gets a set of variable parameters in "param"
        params = {
           'n_estimators': int(params['n_estimators']),
           'learning_rate': "{:.3f}".format(params['learning_rate']),
           'max_depth': int(params['max_depth']),
           'gamma': "{:.3f}".format(params['gamma']),
           'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
           }
        
        # we use this params to create a new LGBM Regressor
        model = xgb.XGBClassifier(objective='binary:logistic',
                        silent=True, nthread=-1, **params)
        
        # and then conduct the cross validation with the same folds as before
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
#        score = -(score.mean() - score.std())
        score = score.mean()
#        print(params, score)
        return score
    
    xgb_space = {
       'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
       'learning_rate': hp.uniform('learning_rate', 1e-4, 0.1),
       'max_depth': hp.quniform('max_depth', 2, 10, 1),
       'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
       'gamma': hp.uniform('gamma', 0.0, 0.5),
    }
    
    # trials will contain logging information
    trials = Trials()
    
    print(t(), 'hyperopt search')
    best = fmin(fn = gb_mse_cv, # function to optimize
              space = xgb_space, 
              algo = anneal.suggest, # optimization algorithm, hyperotp will select its parameters automatically
              max_evals = n_iters, # maximum number of iterations
              trials = trials, # logging
              rstate = np.random.RandomState(random_state) # fixing random state for the reproducibility
             )
    
    clf_ = xgb.XGBClassifier(learning_rate=float(best['learning_rate']), 
                 n_estimators=int(best['n_estimators']), objective='binary:logistic', 
                 silent=True, nthread=-1,  gamma=float(best['gamma']), 
                 colsample_bytree=float(best['colsample_bytree']), max_depth=int(best['max_depth']) )
    print(t(), 'fitting')
    clf_.fit(X_train, y_train)
    print(t(), 'predicting')
    y_pred_test = clf_.predict(X_test)
    return y_pred_test
#    return roc_auc_score(y_test, y_pred_test)

if cfg_load_maps:
    print(t(), 'loading spatial maps from mat')
    X_gigica_raw = []
    X_str_raw = []
    for i in range(num_sub):
        t1 = loadmat(network_path+'gigica_networks/network_'+('%03d' % (i+1))+'.mat')
        t1 = t1['gigica_network']['spatial_maps'][0][0]
        X_gigica_raw.append(np.reshape(t1, (-1, num_comp*num_voxel)))
        
        t1 = loadmat(network_path+'str_networks/network_'+('%03d' % (i+1))+'.mat')
        t1 = np.transpose( t1['str_network']['spatial_maps'][0][0] )
        X_str_raw.append(np.reshape(t1, (-1, num_comp*num_voxel)))
    
    X_gigica_raw = pd.DataFrame(np.squeeze(np.array(X_gigica_raw)))
    X_str_raw = pd.DataFrame(np.squeeze(np.array(X_str_raw).astype(np.float64)))

    with open(outpath+'/networks.pkl', 'wb') as f:
        pickle.dump([X_gigica_raw, X_str_raw], f, protocol=4)

print(t(), 'loading spatial maps from pkl')
with open(outpath+'/networks.pkl', 'rb') as f:
    X_gigica_raw, X_str_raw = pickle.load(f)

scores_gigica = pd.DataFrame(columns=metrics_)
scores_str = pd.DataFrame(columns=metrics_)

#for i in range(repeats):
#    idx_train, idx_test = train_test_split(np.arange(num_sub), test_size=test_size)
#    y_train = y.loc[idx_train]
#    y_test = y.loc[idx_test]
#    
#    selector = NetworkFeaturesTtest(X_gigica_raw.loc[idx_train,:])
#    X_train = selector.fit_transform(X_str_raw.loc[idx_train], y.loc[idx_train])
#    X_test = selector.transform(X_str_raw.loc[idx_test])
#    
#    scores_str.loc[i, 'AUC'] = classify_(X_train, X_test, y_train, y_test)
#    print(t(), 'STR test AUC:', scores_str.loc[i, 'AUC'])
#
#    selector = NetworkFeaturesTtest(X_str_raw.loc[idx_train,:])
#    X_train = selector.fit_transform(X_gigica_raw.loc[idx_train], y.loc[idx_train])
#    X_test = selector.transform(X_gigica_raw.loc[idx_test])
#    
#    scores_gigica.loc[i, 'AUC'] = classify_(X_train, X_test, y_train, y_test)
#    print(t(), 'GIG-ICA test AUC:', scores_gigica.loc[i, 'AUC'])
    
# leave one out 
loo = LeaveOneOut()
for idx_train, idx_test in loo.split(X_gigica_raw):
    print(t(), 'loo', idx_test)
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]
    
    selector = NetworkFeaturesTtest(X_gigica_raw.loc[idx_train,:])
    X_train = selector.fit_transform(X_str_raw.loc[idx_train], y.loc[idx_train])
    X_test = selector.transform(X_str_raw.loc[idx_test])
    
    scores_str.loc[idx_test[0], 'y'] = classify_(X_train, X_test, y_train, y_test)

    selector = NetworkFeaturesTtest(X_str_raw.loc[idx_train,:])
    X_train = selector.fit_transform(X_gigica_raw.loc[idx_train], y.loc[idx_train])
    X_test = selector.transform(X_gigica_raw.loc[idx_test])
    
    scores_gigica.loc[idx_test[0], 'y'] = classify_(X_train, X_test, y_train, y_test)

print(t(), 'GIG-ICA test AUC:', roc_auc_score(y, scores_gigica.loc[:, 'y']))
print(t(), 'STR test AUC:', roc_auc_score(y, scores_str.loc[:, 'y']))

with open(outpath+'/y_loo.pkl', 'wb') as f:
    pickle.dump([scores_gigica, scores_str], f, protocol=4)

#scores = pd.DataFrame(np.transpose(np.vstack( 
#        (scores_gigica.loc[:,'AUC'], scores_str.loc[:,'AUC']) )), columns=['GIG-ICA', 'STR'])
#print(scores.mean())
#
#plt.figure(figsize=(2, 4))
#ax = sns.boxplot(data=scores, notch=True, width=.8)
#ax.set_title('strategy '+str(ttest1_intersect)+str(ttest2_intersect))
#ax.set_ylabel('AUC')

print('Elapsed time:', time.time()-start_time, 's')    
    
    
    
    
