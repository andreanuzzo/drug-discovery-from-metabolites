import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import category_encoders as ce
from pyarrow import feather
import os
import shap
import warnings

from gmem import GeneralMixedEffectsModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

global wdir
wdir = os.getcwd()

def get_gain(model):
      keys = list(model.get_booster().get_score(importance_type='gain').keys())
      gains = np.array(list(model.get_booster().get_score(importance_type='gain').values()))
      indices = np.argsort(gains)[::-1]
      gain_df = pd.DataFrame({'features':[keys[i] for i in indices],
                         'gain': gains[indices]})
      return gain_df
    
def get_fi(importances, features):
        indices = np.argsort(importances)[::-1]
        fi = pd.DataFrame({'features':[features[i] for i in indices],
        'percentage': importances[indices]})
        return fi    

def skfold (abundances, category, savename, clusters, seed=42):   
    
  # Checkpoint save
  print('Saving copy of the data')
  feather.write_feather(abundances, os.path.join(wdir, os.pardir, 'tmp//ML/{}.feather'.format(savename)))
  # #Recall df if necessary `df = feather.read_feather('tmp//df.feather')`
  
  print('Encoding categorical variables with WoEE against nonIBD')
  
  data = abundances
  category = category.iloc[:,0].astype('category')
  
  #encording categorical values with weight of evidence based on IBD/nonIBD
  tmp_data = data
  for i in range(tmp_data.shape[1]):
      if type(tmp_data.iloc[1,i]) is str:
        WOEE = ce.WOEEncoder(cols=[tmp_data.columns[i]])
        WOEE.fit(tmp_data.iloc[:,i], category != 'nonIBD')
        encoded = WOEE.transform(tmp_data.iloc[:,i])
        data = pd.concat([data.drop(tmp_data.columns[i], axis=1), encoded], axis=1)
  
  X = data
  print('Encoding categories')
  labelencoder_y = LabelEncoder()
  Y = labelencoder_y.fit_transform(category)
  target_names = category.unique()
  
  seed = seed
  splits = StratifiedShuffleSplit(test_size = 0.15, random_state=seed)
  for train_index, test_index in splits.split(X, Y):
    X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
    y_train, y_test = Y[train_index], Y[test_index]
    if clusters is None:
      clusters_train, clusters_test = pd.DataFrame(), pd.DataFrame()
    else:
      clusters_train, clusters_test = clusters.loc[train_index,:], clusters.loc[test_index,:]
  
  return X_train, X_test, y_train, y_test, clusters_train, clusters_test

def ML_train_class(X_train, X_test, y_train, y_test, savename=None, seed=42):
  # wdir = os.getcwd()
  # ### Try fitting a bunch of models with default parameters
  scoring = 'f1_weighted'
  modelnames = [
    "Logistic Regression",
    "Nearest Neighbors",
    "Random Forest",
    "Neural Net",
    "Naive Bayes",
    "XGB",
    "One vs rest"]
    
  log = list()

  log.append('training {} models with 10-fold CV and default params'.format(modelnames))
  classifiers = [
      LogisticRegression(class_weight='balanced',
                           multi_class='multinomial',
                           solver='lbfgs', max_iter=500),
      KNeighborsClassifier(n_neighbors = 8),
      RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
      MLPClassifier(alpha=10, random_state=seed,
                    solver='lbfgs', hidden_layer_sizes=[100],
                    max_iter=2000, activation='logistic'),
      GaussianNB(),
      XGBClassifier(random_state=seed, n_jobs=-1, 
                    objective='multi:softmax', num_class=3),
      OneVsRestClassifier(SVC(kernel='linear',probability=True),
                          n_jobs=-1)
      ]

  models = list(zip(modelnames, classifiers))

  results = []
  names = []

  # ### Models train with KFold CV
  log.append("F1-score macro-averaged for cross-validation training (with standard deviation) for each model -")
  for name, model in models:
      kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
      cv_results = cross_val_score(model, X_train, y_train,
                                   cv=kfold, scoring=scoring, n_jobs=-1)
      results.append(cv_results)
      names.append(name)
      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
      log.append(msg)

  # ### Models predictions
  log.append("Saving predictions for each model")
  for name, model in list(models):
      for i, _ in kfold.split(X_train, y_train):
        features = pd.DataFrame(X_train).iloc[i,:]
        cats = pd.DataFrame(y_train).iloc[i,:]
        model.fit(features.values, cats)
      predictions = model.predict(X_test.values)
      pickle.dump(model, open(os.path.join(wdir, os.pardir, 'tmp/ML/{}_{}.sav'.format(savename, name)), 'wb'))
      log.append(name)
      log.append('Accuracy on test set, not averaged: %.3f' % accuracy_score(y_test, predictions))
      score = f1_score(y_test, predictions, average='weighted')
      log.append('Average F1 score, weight-averaged: %.3f' % score)

  return '\n'.join(log)
  
def ML_predict_best(X_train, X_test, y_train, y_test, best_model, tune=False, cv_params=None, force_plot = False):
  
  # wdir = os.getcwd()
  log=list()
  
  try:
    os.makedirs(os.path.join(wdir, os.pardir, 'results'))
    os.makedirs(os.path.join(wdir, os.pardir, 'results/ML'))# will create the directory only if it does not exist
  except FileExistsError:
    pass
  
  loaded_model = pickle.load(open(os.path.join(wdir, os.pardir, best_model), 'rb'))
  if tune is True:
    print('Optimizing {} with automatic RandomSearchCV over {}'\
          .format(best_model, cv_params))
    cv_params = cv_params
    optimized_model = GridSearchCV(loaded_model,
                            cv_params,
                            scoring = 'f1_weighted',
                            cv = 10,
                            n_jobs = -1)
    seed = 42
    optimized_model.fit(X_train.values, y_train)     
    
    loaded_model = pickle.load(open(os.path.join(wdir, os.pardir, best_model), 'rb'))
    y_pred_old = loaded_model.predict(X_test.values)
    y_pred = optimized_model.predict(X_test.values)     
    improvement = f1_score(y_test, y_pred, average="macro")/\
                  f1_score(y_test, y_pred_old, average="macro")
    log.append('New f1 score: ' + str(f1_score(y_test, y_pred, average="macro")))
    log.append('Improvement of {0:.2f}% after optimization'.format(improvement))
    log.append('Predictions from optimized model')
    loaded_model.set_params(**optimized_model.best_params_)
    loaded_model.fit(X_train.values, y_train)
    pickle.dump(loaded_model, 
                open(os.path.join(wdir, os.pardir, 
                'tmp/ML/{}_optimized.sav'.\
                  format(best_model.split('.')[0].split('/')[-1])
                ),'wb')
              )
    
  else:
    log.append('Predictions from chosen model ({})'.format(best_model))
    y_pred = loaded_model.predict(X_test.values)
  return y_pred


def ML_explain(best_model, X_test, category, savename=None, tune=False):

  loaded_model = pickle.load(open(os.path.join(wdir, os.pardir, best_model), 'rb'))
  category = category.iloc[:,0].astype('category')
  log = list()

  if 'XGB' in best_model:

    gain_df = get_gain(loaded_model)
    gain_df.cumsum().plot(drawstyle='steps').\
            axhline(y=0.95*sum(gain_df.gain), label='95% of feature gain',
            xmin=0.01, xmax=0.99, linestyle='dashed')
    gain_df.to_csv(os.path.join(wdir, os.pardir, 
    'results/ML/{}_XGB_gains.csv'.\
                    format(savename)),
             index=False)

    fi = get_fi(loaded_model.feature_importances_, X_test.columns)
    fi = fi[fi.percentage >=0.001]
    fi['features'] = fi['features'].astype('str')

    if tune is True:
      fi.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_XGB_optimized_fi.csv'.\
                    format(savename)),
                           index_label=savename)
    else:
      fi.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_XGB_fi.csv'.\
                    format(savename)),
                           index_label=savename)

    # Shapley model evaluations
    shap.initjs()
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X_test)
    p = shap.summary_plot(shap_values, X_test, class_names=category.unique(), show=False)

    if isinstance(shap_values, (list,)):
      shapi_df = pd.DataFrame(shap_values[0],
                            columns=X_test.columns).T.abs().mean(axis=1)
      for i in range(1,len(category.unique())):
        shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i],
                                                    columns=X_test.columns).T.abs().mean(axis=1)
                             ], axis=1)
      shapi_df.columns = category.unique()
    else:
      shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.abs().mean(axis=1)
      shapi_df.columns = category.unique()[0]

    if tune is True:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_XGB_optimized_shapi.csv'.\
                    format(savename)), index_label=savename)

    else:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_XGB_shapi.csv'.\
                    format(savename)), index_label=savename)

  elif 'Forest' in best_model:
    fi = get_fi(loaded_model.feature_importances_, X_test.columns)
    fi = fi[fi.percentage >=0.001]
    fi['features'] = fi['features'].astype('str')
    fi.to_csv(os.path.join(wdir, os.pardir,
    'results/ML/{}_RF_fi.csv'.\
                    format(savename,
                           index_label=savename)),
             index=False)

    # Shapley model evaluations
    shap.initjs()
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X_test)
    p = shap.summary_plot(shap_values, X_test, class_names=category.unique(), show=False)

    if isinstance(shap_values, (list,)):
      shapi_df = pd.DataFrame(shap_values[0],
                            columns=X_test.columns).T.abs().mean(axis=1)
      for i in range(1,len(category.unique())):
        shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i],
                                                    columns=X_test.columns).T.abs().mean(axis=1)
                             ], axis=1)
      shapi_df.columns = category.unique()
    else:
      shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.abs().mean(axis=1)
      shapi_df.columns = category.unique()[0]

    if tune is True:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_{}_optimized_shapi.csv'.\
                    format(savename,
                           best_model.split('.')[0].lstrip(str([os.path.join(wdir, os.pardir,'tmp/ML/'), savename]))
                          )), index_label=savename)

    else:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_{}_shapi.csv'.\
                    format(savename,
                           best_model.split('.')[0].lstrip(str([os.path.join(wdir, os.pardir,'tmp/ML/'), savename]))
                          )), index_label=savename)
  else:
    shap.initjs()
    log.append('You have selected a {} model. Sit back and relax, it will take a while'.\
          format(best_model.split('.')[0].lstrip(str([os.path.join(wdir, os.pardir,'tmp/ML/'),savename]))))
    explainer = shap.KernelExplainer(loaded_model.predict,
                                     shap.kmeans(X_train, int(np.sqrt(len(X_train)))))
    shap_values = explainer.shap_values(X_test)
    p = shap.summary_plot(shap_values, X_test, class_names=category.unique(), show=False)

    if force_plot is True:
      shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

    if isinstance(shap_values, (list,)):
      shapi_df = pd.DataFrame(shap_values[0],
                            columns=X_test.columns).T.abs().mean(axis=1)
      for i in range(1,len(category.unique())):
        shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i],
                                                    columns=X_test.columns).T.abs().mean(axis=1)
                             ], axis=1)
      shapi_df.columns = category.unique()

    else:
      shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.mean(axis=1).to_frame()
      shapi_df.columns = [savename]

    if tune is True:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_{}_optimized_shapi.csv'.\
                    format(savename,
                           best_model.split('.')[0].lstrip(str([os.path.join(wdir, os.pardir,'tmp/ML/'), savename]))
                          )), index_label=savename)
    else:
      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/ML/{}_{}_shapi.csv'.\
                    format(savename,
                           best_model.split('.')[0].lstrip(str([os.path.join(wdir, os.pardir,'tmp/ML/'), savename]))
                          )), index_label=savename)

  return '\n'.join(log)
  

def GMML(X_train, X_test, y_train, y_test, category, clusters_train=pd.DataFrame(), clusters_test=pd.DataFrame(), estimator=None, random_effects=[], fixed_effects=[], savename='', max_iter = 2, min_iter=1):
  # wdir = os.getcwd()
  log = list()
  seed = 42
  
  if estimator is None:
    estimator = XGBClassifier(random_state=seed, 
                              objective='multi:softprob', 
                              num_class=3, n_jobs=-1)

  #Vanilla model
  if len(random_effects)==0 and len(fixed_effects)==0:
    log.append('\nVanilla model')  
    
    xgb_cv = estimator
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    for i, _ in kfold.split(X_train, y_train):
      features = pd.DataFrame(X_train.iloc[i,:]).\
                            drop(random_effects + \
                                  list(set(fixed_effects)- set(random_effects)), 
                                axis=1)
      cats = pd.DataFrame(y_train).iloc[i,:]
      xgb_cv.fit(features.values, cats)
    
    val_set = X_test.\
                drop(random_effects + \
                      list(set(fixed_effects)- set(random_effects)), 
                    axis=1)
    
    y_pred = xgb_cv.predict(val_set.values)

    name = str(estimator).split('(')[0]
    
    f1_s = np.round(f1_score(y_test,y_pred, average="weighted"), 3)

    log.append('Estimator =' + name + 
         '\nF1 score, weighted = ' + str(f1_s)+
         '\nNo random effects'+ 
         '\nNo fixed effects')
    
    log.append('Saving model')
    name = str(estimator).split('(')[0]
    pickle.dump(xgb_cv, open(os.path.join(wdir,  os.pardir, 'tmp/GMML/{}_no_gm_{}.sav'.format(savename, name)), 'wb'))
    log.append('Saving feature importances\n')    
    fi = get_fi(xgb_cv.feature_importances_, val_set.columns)
    fi = fi[fi.percentage >=0.001]
    fi['features'] = fi['features'].astype('str')
    fi.to_csv(os.path.join(wdir, os.pardir, 'results/GMML/{}_{}_fi.csv'.\
                        format(savename, name)),
                               index_label=savename)
  #GeneralMixedEffectsModel XGBClassifier with clusters
  if len(clusters_train)>0 and len(random_effects)==0:
    log.append('\nGeneralized mixed effects model, only clusters')  
    
    relm = GeneralMixedEffectsModel(estimator=estimator, 
                                    cv=5, gll_early_stop_threshold=0.1,
                                    max_iterations=max_iter, 
                                    min_iterations=min_iter, 
                                    n_jobs=-1, 
                                    verbose=True)

    features = X_train.\
                drop(random_effects + \
                      list(set(fixed_effects) - set(random_effects)), 
                    axis=1)

    Z_train = np.ones((len(features), 1))
    cats = y_train #transform_probability(diagnosis)[train_index]
    
    relm.fit(features.values, Z_train, clusters_train.iloc[:,0], cats)

    val_set = X_test.\
                drop(random_effects + \
                      list(set(fixed_effects)- set(random_effects)), 
                    axis=1)

    Z_val = np.ones((len(X_test), 1))

    y_pred = np.round(relm.predict(val_set.values, Z_val, clusters_test.iloc[:,0]))
    y_pred[y_pred>y_test.max()] = y_test.max()
    y_pred[y_pred<0] = 0
    pred_train = relm.predict(features.values, Z_train, clusters_train.iloc[:,0])

    name = str(estimator).split('(')[0]
    f1_s = np.round(f1_score(y_test, y_pred, average="weighted"), 3) 
    cs = clusters_train.iloc[:,0].name

    log.append('GMME with '+ name+ 
          '\nF1 score, weighted = '+ str(f1_s) +
          '\nclusters = '+ cs +
          '\nNo random effects'+
          '\nNo fixed effects')
    
    log.append('Saving model')
    pickle.dump(relm, open(os.path.join(wdir, os.pardir, 'tmp/GMML/{}_gmme_cl_{}.sav'.format(savename, name)), 'wb'))
    log.append('Saving feature importances\n')
    fi = get_fi(relm.estimator_.feature_importances_, val_set.columns)
    fi = fi[fi.percentage >=0.001]
    fi['features'] = fi['features'].astype('str')
    fi.to_csv(os.path.join(wdir, os.pardir,
    'results/GMML/{}_gmme_{}_fi.csv'.\
                        format(savename, name)),
                               index_label=savename)

    if 'XGB' in str(estimator):
    # Shapley model evaluations
      shap.initjs()
      explainer = shap.TreeExplainer(relm.estimator_)
      shap_values = explainer.shap_values(X_test.values)

      if isinstance(shap_values, (list,)):
        shapi_df = pd.DataFrame(shap_values[0], 
                              columns=X_test.columns).T.abs().mean(axis=1)
        for i in range(1,len(category.iloc[:,0].unique())):
          shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i], 
                                                      columns=X_test.columns).T.abs().mean(axis=1)
                               ], axis=1)
        shapi_df.columns = category.iloc[:,0].unique()
      else:
        shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.abs().mean(axis=1)
        shapi_df.columns = category.iloc[:,0].unique()[0]

      shapi_df.to_csv(os.path.join(wdir, os.pardir,
      'results/GMML/{}_gmme_{}_shapi.csv'.\
                        format(savename, name)),
                               index_label=savename)
  #GeneralMixedEffectsModel XGBClassifier with clusters, categorical variables as random effects
  if len(clusters_train)>0 and len(random_effects)>0 and len(fixed_effects)==0:
      grid = {'num_class' :len(category.iloc[:,0].unique())}
      log.append('\nGeneralized mixed effects model, clusters and random effects')  
      gmme_re = GeneralMixedEffectsModel(estimator=estimator.set_params(**grid), 
                                      cv=5, gll_early_stop_threshold=0.1,
                                      max_iterations=max_iter, min_iterations=min_iter, 
                                      n_jobs=-1, verbose=True)

      features = X_train.\
                    drop(random_effects + \
                          list(set(fixed_effects)- set(random_effects)), 
                        axis=1)

      Z_train = np.concatenate([np.ones((len(features), 1)), (features.values)], 
                                 axis=1)

      cats = y_train

      gmme_re.fit(features.values, Z_train, clusters_train.iloc[:,0], cats)

      val_set = X_test.\
                  drop(random_effects + \
                        list(set(fixed_effects)- set(random_effects)), 
                      axis=1)

      Z_val = np.concatenate([np.ones((len(X_test), 1)), 
                             (val_set.values)], 
                              axis=1)

      y_pred = np.round(gmme_re.predict(val_set.values, Z_val, clusters_test.iloc[:,0])) 
      y_pred[y_pred>y_test.max()] = y_test.max()
      y_pred[y_pred<0] = 0

      name = str(estimator).split('(')[0]
      f1_s = np.round(f1_score(y_test, y_pred, average="weighted"), 3)
      cs = clusters_train.iloc[:,0].name
      re = ','.join([str(i) for i in random_effects])

      log.append('GMME with '+ name+ 
            '\nF1 score, weighted = '+ str(f1_s)+
            '\nclusters = '+ cs+
            '\nRandom effects = '+ re +
            '\nNo fixed effects')
      
      log.append('Saving model')
      pickle.dump(gmme_re, open(os.path.join(wdir, os.pardir, 'tmp/GMML/{}_gmme_re_{}.sav'.format(savename, name)), 'wb'))
      log.append('Saving feature importances\n')
      fi = get_fi(gmme_re.estimator_.feature_importances_, val_set.columns)
      fi = fi[fi.percentage >=0.001]
      fi['features'] = fi['features'].astype('str')
      fi.to_csv(os.path.join(wdir, os.pardir,'results/GMML/{}_gmme_re_{}_fi.csv'.\
                      format(savename, name)), index_label=savename)
      
      if 'XGB' in str(estimator):
    # Shapley model evaluations
        shap.initjs()
        explainer = shap.TreeExplainer(gmme_re.estimator_)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, (list,)):
          shapi_df = pd.DataFrame(shap_values[0], 
                                columns=X_test.columns).T.abs().mean(axis=1)
          for i in range(1,len(category.iloc[:,0].unique())):
            shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i], 
                                                        columns=X_test.columns).T.abs().mean(axis=1)
                                 ], axis=1)
          shapi_df.columns = category.iloc[:,0].unique()
        else:
          shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.abs().mean(axis=1)
          shapi_df.columns = category.iloc[:,0].unique()[0]

        shapi_df.to_csv(os.path.join(wdir, os.pardir,'results/GMML/{}_gmme_re_{}_shapi.csv'.\
                        format(savename, name)),
                               index_label=savename)
  #GeneralMixedEffectsModel XGBClassifier with clusters, categorical variables as mixed effects
  if len(clusters_train)>0 and len(random_effects)>0 and len(fixed_effects)>0:
    log.append('\nGeneralized mixed effects model, clusters, random effects and fixed effects')  
    grid = {'num_class' :len(category.iloc[:,0].unique())}
    gmme_me = GeneralMixedEffectsModel(estimator=estimator.set_params(**grid), 
                                    cv=5, gll_early_stop_threshold=0.1,
                                    max_iterations=max_iter, min_iterations=min_iter, 
                                    n_jobs=-1, verbose=True)

    features = X_train
    Z_train = np.concatenate([np.ones((len(features), 1)), 
                              (features.values)], 
                               axis=1)

    cats = y_train
    gmme_me.fit(features.values, Z_train, clusters_train.iloc[:,0], cats)

    Z_val = np.concatenate([np.ones((len(X_test), 1)), (X_test.values)], 
                               axis=1)
                               
    y_pred = np.round(gmme_me.predict(X_test.values, Z_val, clusters_test.iloc[:,0]))
    y_pred[y_pred>y_test.max()] = y_test.max()
    y_pred[y_pred<0] = 0

    name = str(estimator).split('(')[0]
    f1_s = np.round(f1_score(y_test, y_pred, average="weighted"), 3)
    cs = clusters_train.iloc[:,0].name
    re = ','.join([str(i) for i in random_effects])
    fe = ','.join([str(i) for i in fixed_effects])

    log.append('GMME with '+ name+
          '\nF1 score, weighted = '+ str(f1_s)+
          '\nclusters = '+ cs+ 
          '\nRandom effects = '+ re+
          '\nFixed effects = '+ fe)

    log.append('Saving model')
    pickle.dump(gmme_me, open(os.path.join(wdir, os.pardir, 'tmp/GMML/{}_gmme_me_{}.sav'.format(savename, name)), 'wb'))
    
    log.append('Saving feature importances')
    fi = get_fi(gmme_me.estimator_.feature_importances_, X_test.columns)
    fi = fi[fi.percentage >=0.001]
    fi['features'] = fi['features'].astype('str')
    fi.to_csv(os.path.join(wdir, os.pardir,'results/GMML/{}_gmme_me{}_fi.csv'.\
                    format(savename, name)),index_label=savename)
    
    if 'XGB' in str(estimator):
    # Shapley model evaluations
      shap.initjs()
      explainer = shap.TreeExplainer(gmme_me.estimator_)
      shap_values = explainer.shap_values(X_test)
      
      if isinstance(shap_values, (list,)):
        shapi_df = pd.DataFrame(shap_values[0], 
                              columns=X_test.columns).T.abs().mean(axis=1)
        for i in range(1,len(category.iloc[:,0].unique())):
          shapi_df = pd.concat([shapi_df,pd.DataFrame(shap_values[i], 
                                                      columns=X_test.columns).T.abs().mean(axis=1)
                               ], axis=1)
        shapi_df.columns = category.iloc[:,0].unique()
      else:
        shapi_df = pd.DataFrame(shap_values,columns=X_test.columns).T.abs().mean(axis=1)
        shapi_df.columns = category.iloc[:,0].unique()[0]

      shapi_df.to_csv(os.path.join(wdir, os.pardir,'results/GMML/{}_gmme_me_{}_shapi.csv'.\
                        format(savename, name)),
                               index_label=savename)
  print('\n'.join(log))

  return y_pred

