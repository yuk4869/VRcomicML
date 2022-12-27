# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 06:55:12 2021

@author: sakam
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import scipy

def Data_Remake(df):#コマごとにクエリしてデータ抽出
    remade = pd.DataFrame(columns = PanelColumns)
    for page,panel in zip(term['Page'], term['Panel']):
        new = pd.DataFrame(columns = df.columns)
        queried = df.query('Page == @page & Panel == @panel')
        if len(queried) <= 1:continue
        new = new.append(queried.loc[:, queried.columns.str.contains('Time|_Sum|_L')].sum(),ignore_index = True)
        new['GazeMove_L'] = new['GazeMove_L'] / new['Time']
        new.loc[0, queried.columns.str.contains('_Mean')] \
            = (queried['FrameCount'].dot(queried.loc[:, queried.columns.str.contains('_Mean')]) / queried['FrameCount'].sum()).T
        new['BLINK_D_Mean'] = (queried['BLINK_N'].dot(queried['BLINK_D_Mean'])) / queried['BLINK_N'].sum()
        new.loc[0,queried.columns.str.contains('_N')] \
            = queried.loc[:, queried.columns.str.contains('_N')].sum() / new.Time[0]
        new.loc[0, queried.columns.str.contains('_Max')] \
            = queried.loc[:, queried.columns.str.contains('_Max')].max()
        new.loc[0, queried.columns.str.contains('_Min')] \
            = queried.loc[:, queried.columns.str.contains('_Min')].min()
        new['FIX_D_Mean'] = (queried['FIX_N'].dot(queried['FIX_Mean'])) / queried['FIX_N'].sum()
        new['Panel_Move_Min'] = queried['Panel_Move'].min()
        new['Panel_Move_Mean'] = queried['Panel_Move'].mean()
        new['Panel_Move_Max'] = queried['Panel_Move'].max()
        new['Tile_Move_Min'] = queried['Tile_Move'].min()
        new['Tile_Move_Mean'] = queried['Tile_Move'].mean()
        new['Tile_Move_Max'] = queried['Tile_Move'].max()
        new.rename(columns = {
        'FIX_Min':'FIX_D_Min', 
        'FIX_Max':'FIX_D_Max', 
        }, inplace = True)
        
#        new.loc[0, queried.columns.str.contains('_Move')] \
#            = queried.loc[:, queried.columns.str.contains('_Move')].mean()
        new['Page'] = page
        new['Panel'] = panel
        new['IsDifficult']  = queried['IsDifficult'].max()
        new['IsUnderstand'] = queried['IsUnderStand'].max()
        new['TextNum'] = queried.TextNum.mode()
        new = new.dropna(axis = 1)
        remade = remade.append(new.reindex(columns = PanelColumns))
    remade.fillna(0, inplace = True)
    return remade

def Normalize(df):
    temp = pd.DataFrame(scipy.stats.zscore(df),columns = df.columns)
    temp.loc[:, temp.columns.str.contains('_Move')] = df.loc[:, df.columns.str.contains('_Move')]
    return temp

def split_for_LOO(all_data, ID):
    train = all_data.query('ID != @ID').drop('ID', axis = 1)
    test = all_data.query('ID == @ID').drop('ID', axis = 1)
    return train.drop('IsDifficult',axis = 1), test.drop('IsDifficult', axis = 1),\
         train.IsDifficult.astype('int'), test.IsDifficult.astype('int')

def predict_each(X_train,Y_train, X_test, clf):
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)

def Predict(X_train,Y_train, X_test, SVM, RF, i, _target):
    global prior_Importances
    global addHeat_Importances
    global addCon_Importances
    global all_Importances
    _X_train = X_train.drop('TextNum', axis =1)
    _X_test = X_test.drop('TextNum', axis =1)
    prior = np.logical_not(_X_train.columns.str.contains('HM|VGC|Panel|Tile'))
    add_Heat =          prior + _X_train.columns.str.contains('HM')
    add_Cross =         prior + _X_train.columns.str.contains('VGC')


    Y_BL= predict_each(np.array(X_train['TextNum']).reshape(-1, 1),\
                       Y_train, np.array(X_test['TextNum']).reshape(-1, 1), SVM)
    print('prior')
    Y_SVM_prior = predict_each(_X_train.loc[:,prior],\
                               Y_train, _X_test.loc[:,prior], SVM)
    Y_SVM_add_Heat = predict_each(_X_train.loc[:,add_Heat],\
                                  Y_train, _X_test.loc[:,add_Heat], SVM)
    Y_SVM_add_cross = predict_each(_X_train.loc[:,add_Cross],\
                                   Y_train, _X_test.loc[:,add_Cross], SVM)
    allfeature = prior + _X_train.columns.str.contains('HM') +\
                             _X_train.columns.str.contains('VGC')+\
                             _X_train.columns.str.contains('Tile')  +\
                           _X_train.columns.str.contains('Panel')
    Y_SVM_all = predict_each(_X_train, Y_train,  _X_test.loc[:,allfeature] , SVM)

    Y_RF_prior = predict_each(_X_train.loc[:,prior],\
                               Y_train, _X_test.loc[:,prior], RF)
    Y_RF_add_Heat = predict_each(_X_train.loc[:,add_Heat],\
                                  Y_train, _X_test.loc[:,add_Heat], RF)
    Y_RF_add_cross = predict_each(_X_train.loc[:,add_Cross],\
                                   Y_train, _X_test.loc[:,add_Cross], RF)
    Y_RF_all = predict_each(_X_train, Y_train, _X_test, RF)
    
    return PanelPredict(prior, Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all)

def PanelPredict(prior, Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all):
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
    return PredictEach(prior,Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, \
                Y_SVM_add_PanelMove, Y_SVM_add_TileMove, Y_RF_add_PanelMove, Y_RF_add_TileMove,\
                Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all)
    return [Y_BL, \
            Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, \
            Y_SVM_add_PanelMove, Y_SVM_add_TileMove, Y_SVM_all,\
            Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, \
            Y_RF_add_PanelMove, Y_RF_add_TileMove, Y_RF_all]
        
def PredictEach(prior,Y_train, _X_train, _X_test, SVM, RF, Y_BL, Y_SVM_prior, \
                Y_SVM_add_PanelMove, Y_SVM_add_TileMove, Y_RF_add_PanelMove, Y_RF_add_TileMove,\
                Y_SVM_add_Heat, Y_SVM_add_cross, Y_SVM_all, \
                Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, Y_RF_all):
    add_PanelMat = prior + _X_train.columns.str.contains('Panel_Move')
    add_TileMat = prior + _X_train.columns.str.contains('Tile_Move')
    HM_VGC =            prior + _X_train.columns.str.contains('HM') + _X_train.columns.str.contains('VGC')
    HM_PM =             prior + _X_train.columns.str.contains('HM') + add_PanelMat
    HM_TM =             prior + _X_train.columns.str.contains('HM') + add_TileMat
    HM_VGC_PM =         HM_VGC + add_PanelMat
    HM_VGC_TM =         HM_VGC + add_TileMat
    HM_PM_TM =          HM_PM + HM_TM
    VGC_PM =            _X_train.columns.str.contains('VGC') + add_PanelMat
    VGC_TM =            _X_train.columns.str.contains('VGC') + add_TileMat
    VGC_PM_TM =         VGC_PM + VGC_TM
    PM_TM =             add_PanelMat + add_TileMat
    #add each
    Y_SVM_HM_VGC = predict_each(_X_train.loc[:,HM_VGC],\
                                   Y_train, _X_test.loc[:,HM_VGC], SVM)
    Y_SVM_HM_PM = predict_each(_X_train.loc[:,HM_PM],\
                                   Y_train, _X_test.loc[:,HM_PM], SVM)
    Y_SVM_HM_TM = predict_each(_X_train.loc[:,HM_TM],\
                                   Y_train, _X_test.loc[:,HM_TM], SVM)
    Y_SVM_VGC_PM= predict_each(_X_train.loc[:,VGC_PM],\
                                   Y_train, _X_test.loc[:,VGC_PM], SVM)
    Y_SVM_VGC_TM= predict_each(_X_train.loc[:,VGC_TM],\
                                   Y_train, _X_test.loc[:,VGC_TM], SVM)
    Y_SVM_PM_TM= predict_each(_X_train.loc[:,PM_TM],\
                                   Y_train, _X_test.loc[:,PM_TM], SVM)
        
    Y_SVM_HM_VGC_PM = predict_each(_X_train.loc[:,HM_VGC_PM],\
                                   Y_train, _X_test.loc[:,HM_VGC_PM], SVM)
    Y_SVM_HM_VGC_TM= predict_each(_X_train.loc[:,HM_VGC_TM],\
                                   Y_train, _X_test.loc[:,HM_VGC_TM], SVM)
    Y_SVM_HM_PM_TM = predict_each(_X_train.loc[:,HM_PM_TM],\
                                   Y_train, _X_test.loc[:,HM_PM_TM], SVM)
    Y_SVM_VGC_PM_TM= predict_each(_X_train.loc[:,VGC_PM_TM],\
                                   Y_train, _X_test.loc[:,VGC_PM_TM], SVM)
    return [Y_BL, \
            Y_SVM_prior, Y_SVM_add_Heat, Y_SVM_add_cross, \
            Y_SVM_add_PanelMove, Y_SVM_add_TileMove, Y_SVM_all,\
            Y_RF_prior, Y_RF_add_Heat, Y_RF_add_cross, \
            Y_RF_add_PanelMove, Y_RF_add_TileMove, Y_RF_all, Y_SVM_HM_VGC, \
            Y_SVM_HM_PM, Y_SVM_HM_TM, Y_SVM_HM_VGC_PM, Y_SVM_HM_VGC_TM,\
            Y_SVM_VGC_PM, Y_SVM_VGC_TM, Y_SVM_VGC_PM_TM, Y_SVM_PM_TM ,Y_SVM_HM_PM_TM]
    
def After_fit(test, pred, target, title):
    CM(test, pred, target, title)
    report = pd.DataFrame(classification_report(test,pred,output_dict=True))
    #print(report)
    return [report.loc['precision','accuracy'], 
                 report.loc['precision','macro avg'], 
                 report.loc['recall','macro avg'], 
                 report.loc['f1-score','macro avg'], 
                 report.loc['support','1'], 
                 report.loc['support','0'],
                 report.loc['support', 'macro avg']]

def After_fin_target(aggregates):
    _allAgg = pd.DataFrame()
    _n = 0
    for _aggregate in aggregates:
        _aggregate['result'] = _aggregate.drop('result', axis = 1).mean(axis = 1)
        _aggregate['min'] = _aggregate.min(axis = 1)
        _aggregate['max'] = _aggregate.max(axis = 1)
        _aggregate['std'] = _aggregate.std(axis = 1)
        _aggregate.loc['resultName'] = resultName[_n]
        if _n == 0:
            _allAgg = _aggregate
        else:
            _allAgg = pd.concat([_allAgg,_aggregate])
        _aggregate.to_csv(path_or_buf = saveDir  + target + '/'+resultName[_n]+".csv")  
        _n += 1
    return _allAgg

def Add_All_Result(_result):
    global allResult
    indexList = ['acc', 'precision', 'recall', 'f1-score']
    for _index in indexList:
        allResult[target + _index] = np.array(_result.loc[_index,'result'])
    allResult[target+'std'] = np.array(_result.loc['f1-score','std'])

def CM(test, pred, target, title):
    cm = confusion_matrix(test, pred)
    plt.rcParams["font.size"] = 18
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(title + ' ' + target)
    plt.savefig("plot/General/"+ target + '/' + title + ".png")
    plt.show()


def Aggregate():
    return pd.DataFrame(index = [ 'acc',
                            'precision',
                            'recall',
                            'f1-score',
                            'TRUE',
                            'FALSE',
                            'data num'
                            ],
                 columns = ['result'])

saveDir = 'data/classification report/General/'
params = [
{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
]
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]


term = pd.read_csv('data/TextNum.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page').astype('int')
term_Page = term.groupby('Page').sum().reset_index().drop('Panel', axis = 1)
resultName = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                      'SVM_add_PanelMove','SVM_add_TileMove', 'SVM_all',\
                      'RF_prior', 'RF_add_Heat', 'RF_add_cross', \
                      'RF_add_PanelMove', 'RF_add_TileMove', 'RF_all',\
                      'Y_SVM_HM_VGC','Y_SVM_HM_PM', 'Y_SVM_HM_TM', 'Y_SVM_HM_VGC_PM', 'Y_SVM_HM_VGC_TM',\
            'Y_SVM_VGC_PM', 'Y_SVM_VGC_TM', 'Y_SVM_VGC_PM_TM', 'Y_SVM_PM_TM' ,'Y_SVM_HM_PM_TM']             
PanelColumns = [#'PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max', 
                'FIX_D_Min', 'FIX_D_Mean','FIX_D_Max', 'FIX_N',
                'BLINK_D_Min', 'BLINK_D_Mean','BLINK_D_Max', 'BLINK_N',
                'Panel_Move_Min','Panel_Move_Mean','Panel_Move_Max',
                'Tile_Move_Min','Tile_Move_Mean','Tile_Move_Max',
                'GazeMove_L', 'Time', 
                'HM_Min', 'HM_Mean', 'HM_Max',  
                'VGC_Min', 'VGC_Mean', 'VGC_Max', 
                'TextNum', 'IsDifficult']

clf_RF = RandomForestClassifier(max_features = 3, class_weight = "balanced", n_jobs=-1, random_state=912)
clf_SVM = GridSearchCV( SVC(class_weight="balanced", probability = True), params, n_jobs=-1,cv=3,scoring="f1_macro")
DropList = []
allResult = pd.DataFrame(index = resultName)
target = 'Panel'

prior_Importances= []
addHeat_Importances= []
addCon_Importances= []
all_Importances = []

SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]

def Main():
    aggregate = [Aggregate() for i in range(len(resultName))]
    eye = pd.DataFrame()
    for ID in range(2,26,1):
        if ID in SkipID:continue
        described_data = pd.read_csv('data/describe/'+target+'/' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        described_data = described_data.dropna()
        described_data = pd.merge(described_data, term, how = 'inner', on = ['Page','Panel']).astype('float') 
        described_data.rename(columns = {
        'DiffCrossPos_Min':'VGC_Min', 
        'DiffCrossPos_Mean':'VGC_Mean', 
        'DiffCrossPos_Max':'VGC_Max',
        'Heatmap_Min':'HM_Min',
        'Heatmap_Mean':'HM_Mean',
        'Heatmap_Max':'HM_Max'}, inplace = True)
        temp = pd.DataFrame()
        described_data = Data_Remake(described_data)
        described_data = described_data.reindex(columns = PanelColumns)
        temp = Normalize(described_data) 
        temp['IsDifficult'] = np.array(described_data['IsDifficult'])
        temp['ID'] = ID
        eye = pd.concat([eye, temp])
    ###ここからLOO
    for ID in range(2, 26, 1):
        if ID in SkipID: continue
        X_train, X_test, Y_train, Y_test = split_for_LOO(eye, ID)    
        results = Predict(X_train, Y_train, X_test, clf_SVM, clf_RF, ID, target)
        n = 0
        for result in results:
            aggregate[n][ID] = After_fit(Y_test, result, target, resultName[n])
            n += 1
    Add_All_Result(After_fin_target(aggregate))

    allResult.to_csv(path_or_buf = saveDir +"GeneralResultPanel.csv")

if __name__ == "__main__":
    Main()
