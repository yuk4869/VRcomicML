# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:24:59 2022

@author: sakam
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
import statistics as st
import scipy.stats as stpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def Standardization(train, test):
    _train = pd.DataFrame()
    _test = pd.DataFrame()
    for column in train.columns:
        sig = st.pstdev(train[column])
        mean = train[column].mean()
        _train[column] = (train[column] - mean) / sig
        _test[column] = (test[column] - mean) / sig
    return _train, _test

def split_for_time(columns, df):
    Train_Data = df.query('Page % 3 != 0')
    Test_Data = df.query('Page % 3 == 0')
    #X_train, X_test, Y_train, Y_test
    print('a')
    return Train_Data[columns].drop('IsDifficult', axis =1), Test_Data[columns].drop('IsDifficult', axis =1), Train_Data.IsDifficult.astype('int'), Test_Data.IsDifficult.astype('int')

def predict_each(X_train,Y_train, X_test, clf):
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)

def Predict(X_train,Y_train, X_test, SVM, RF, i):
    global prior_Importances
    global addHeat_Importances
    global addCon_Importances
    global all_Importances
    
    _X_train = X_train.drop('TextNum', axis =1)
    _X_test = X_test.drop('TextNum', axis =1)
    prior = np.logical_not(_X_train.columns.str.contains('HM|VGC|Panel|Tile'))
    add_Heat = np.logical_not(_X_train.columns.str.contains('VGC'))
    add_Cross = np.logical_not(_X_train.columns.str.contains('HM'))
    

    Y_BL= predict_each(np.array(X_train['TextNum']).reshape(-1, 1), Y_train, np.array(X_test['TextNum']).reshape(-1, 1), SVM)
    Y_SVM_prior = predict_each(_X_train.loc[:,prior],\
                               Y_train, _X_test.loc[:,prior], SVM)
    Y_SVM_add_Heat = predict_each(_X_train.loc[:,add_Heat],\
                                  Y_train, _X_test.loc[:,add_Heat], SVM)
    Y_SVM_add_cross = predict_each(_X_train.loc[:,add_Cross],\
                                   Y_train, _X_test.loc[:,add_Cross], SVM)
    Y_SVM_all = predict_each(_X_train, Y_train, _X_test, SVM)

    Y_RF_add_Heat = predict_each(_X_train.loc[:,add_Heat],\
                                  Y_train, _X_test.loc[:,add_Heat], RF)
    Y_RF_add_cross = predict_each(_X_train.loc[:,add_Cross],\
                                   Y_train, _X_test.loc[:,add_Cross], RF)
    Y_RF_all = predict_each(_X_train, Y_train, _X_test, RF)
            #Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all,Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, \
    return PanelPredict(prior, Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                 Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all)

def PanelPredict(prior, Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                 Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all):
    add_PanelMove = prior + _X_train.columns.str.contains('Panel_Move')
    add_TileMove = prior + _X_train.columns.str.contains('Tile_Move')
    Y_SVM_add_PanelMove = predict_each(_X_train.loc[:,add_PanelMove],\
                                   Y_train, _X_test.loc[:,add_PanelMove], SVM)
    Y_SVM_add_TileMove = predict_each(_X_train.loc[:,add_TileMove],\
                                   Y_train, _X_test.loc[:,add_TileMove], SVM)
    Y_RF_add_PanelMove = predict_each(_X_train.loc[:,add_PanelMove],\
                                   Y_train, _X_test.loc[:,add_PanelMove], RF)
    Y_RF_add_TileMove = predict_each(_X_train.loc[:,add_TileMove],\
                                   Y_train, _X_test.loc[:,add_TileMove], RF)
    return [Y_BL, \
            Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, \
            Y_SVM_add_PanelMove, Y_SVM_add_TileMove, Y_SVM_all,\
            Y_RF_add_Heat, Y_RF_add_cross, \
            Y_RF_add_PanelMove, Y_RF_add_TileMove, Y_RF_all]
def After_fit(test, pred, target, ID, title):
    if 'SVM' in title:
        _saveDir = saveDir +'SVM/'
    elif 'RF' in title:
        _saveDir = saveDir + 'RF/'
    else: 
        _saveDir = saveDir + 'BL/'
    CM(test, pred, target, ID, title)
    report = pd.DataFrame(classification_report(test,pred,output_dict=True))
    #report.to_csv(path_or_buf = _saveDir + target + '/'+ title + str(ID) + ".csv" ,index = False)
    return [report.loc['precision','accuracy'], 
                 report.loc['precision','macro avg'], 
                 report.loc['recall','macro avg'], 
                 report.loc['f1-score','macro avg'], 
                 report.loc['support','1'], 
                 report.loc['support','0'],
                 report.loc['support', 'macro avg']]

def CM(test, pred, target, ID, title):
    if 'SVM' in title:
        _saveDir = 'plot/SVM/'
    elif 'RF' in title:
        _saveDir = 'plot/RF/'
    else: 
        _saveDir = 'plot/BL/'
    cm = confusion_matrix(test, pred)
    plt.rcParams["font.size"] = 18
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(title + ' ' + target + ' ' + str(ID))
    plt.savefig(_saveDir + target + '/'+ title + str(ID) +".png")
    plt.show()

    
def After_fin_target(aggregates):
    _allAgg = pd.DataFrame()
    _n = 0
    for _aggregate in aggregates:
        _aggregate['average'] = _aggregate.mean(axis = 1)
        _aggregate['min'] = _aggregate.min(axis = 1)
        _aggregate['max'] = _aggregate.max(axis = 1)
        _aggregate.loc['resultName'] = resultName[_n]
        if _n == 0:
            _allAgg = _aggregate
        else:
            _allAgg = pd.concat([_allAgg,_aggregate])
        _aggregate.to_csv(path_or_buf = saveDir + 'Aggregate/' + target + '/'+resultName[_n]+".csv")  
        _n += 1
    return _allAgg

def Add_All_Result(_result):
    indexList = ['acc', 'precision', 'recall', 'f1-score']
    for _index in indexList:
        allResultAve[target + _index] = np.array(_result.loc[_index,'average'])
        allResultMin[target + _index] = np.array(_result.loc[_index,'min'])
        allResultMax[target + _index] = np.array(_result.loc[_index,'max'])

def Aggregate():
    return pd.DataFrame(index = [ 'acc',
                            'precision',
                            'recall',
                            'f1-score',
                            'TRUE',
                            'FALSE',
                            'data num'
                            ],
                 columns = ['average'])

skipflag = False
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
params = [
{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
]
Eval = 'Difficult'
targets = ['Fixation']

resultName = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                      'SVM_add_PanelMove','SVM_add_TileMove', 'SVM_all',\
                      'RF_add_Heat', 'RF_add_cross', \
                      'RF_add_PanelMove', 'RF_add_TileMove', 'RF_all']             
FixationColumns = [#'PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max', 
                   'FIX_D', 'GazeMove_L', 
                   'HM_Min', 'HM_Mean', 'HM_Max',
                   'Panel_Move', 'Tile_Move',
                   'VGC_Min', 'VGC_Mean', 'VGC_Max', 
                   'TextNum', 'IsDifficult']

sm = SMOTE(random_state=912)
saveDir = 'data/classification report/'
clf_SVM = GridSearchCV( SVC(class_weight="balanced", probability = True), params, n_jobs=-1,cv=3,scoring="f1_macro")
clf_RF = RandomForestClassifier(max_features = 3,  class_weight = "balanced", n_jobs=-1, random_state=912)
              #'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', 'SVM_all'
term = pd.read_csv('data/TextNum.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
DropList = []
allResultAve = pd.DataFrame(index = resultName)
allResultMin = pd.DataFrame(index = resultName)
allResultMax = pd.DataFrame(index = resultName)

prior_Importances= []
addHeat_Importances= []
addCon_Importances= []
all_Importances = []

SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]

for target in targets:
    aggregate = [Aggregate() for i in range(len(resultName))]
    skipList = []
    i = 0
    f1 = []
    AllROC = pd.DataFrame()
    for ID in range(2, 26, 1):
        if ID in SkipID: continue
        i += 1    
        described_data = pd.read_csv('data/describe/'+target+'/' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        described_data = described_data.dropna()
        described_data = pd.merge(described_data, term, how = 'inner', on = ['Page','Panel'])
        described_data.rename(columns = {
        'DiffCrossPos_Min':'CE_Min', 
        'DiffCrossPos_Mean':'CE_Mean', 
        'DiffCrossPos_Max':'CE_Max',
        'Heatmap_Min':'HM_Min',
        'Heatmap_Mean':'HM_Mean',
        'Heatmap_Max':'HM_Max'}, inplace = True)
        Y_df = described_data.IsDifficult.astype('int')
        X_df = described_data
        try:
            X = X_df.reindex(columns = FixationColumns)
            X_train, X_test, Y_train, Y_test = train_test_split(X.drop('IsDifficult', axis =1), Y_df, test_size=0.3, stratify=Y_df, random_state=42)
            X_train, X_test = Standardization(X_train, X_test)
            X_train, Y_train = sm.fit_sample(X_train, Y_train)
            skipflag = False
        except: 
            print(str(ID) + ' ' + target + ' skiped')
            skipList.append(i)
            skipflag = True
            #continue  
        if not skipflag:
            results = Predict(X_train, Y_train, X_test, clf_SVM, clf_RF, i)
            n = 0
            for result in results:
                aggregate[n][ID] = After_fit(Y_test, result, target, ID, resultName[n])
                n += 1
        print(str(ID) + target)
    allAggregate = After_fin_target(aggregate)
    Add_All_Result(allAggregate)

allResultAve.to_csv(path_or_buf = saveDir +"PersonalizedFixatonResultAve.csv")
print(allResultAve.loc['RF_all','Fixationf1-score'])

