# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:06:28 2022

@author: sakam
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def getList():
    Pn_Gene = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                          'SVM_add_PanelMove','SVM_add_TileMove', 'SVM_all',\
                          'RF_prior', 'RF_add_Heat', 'RF_add_cross', \
                          'RF_add_PanelMove', 'RF_add_TileMove', 'RF_all',\
                          'Y_SVM_HM_VGC','Y_SVM_HM_PM', 'Y_SVM_HM_TM', 'Y_SVM_HM_VGC_PM', 'Y_SVM_HM_VGC_TM',\
                          'Y_SVM_VGC_PM', 'Y_SVM_VGC_TM', 'Y_SVM_VGC_PM_TM', 'Y_SVM_PM_TM' ,'Y_SVM_HM_PM_TM']      
    Pg_Gene = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                          'SVM_add_UpperPanel', 'SVM_add_UnderPanel', 'SVM_add_CorectPanel',\
                          'SVM_add_UpperTile', 'SVM_add_UnderTile', 'SVM_add_CorectTile',\
                          'SVM_PanelMat_all', 'SVM_TileMat_all', 'SVM_all',\
                          'RF_prior', 'RF_add_Heat', 'RF_add_cross', \
                          'RF_add_UpperPanel', 'RF_add_UnderPanel', 'RF_add_CorectPanel', \
                          'RF_add_UpperTile', 'RF_add_UnderTile', 'RF_add_CorectTile',\
                          'RF_PanelMat_all', 'RF_TileMat_all', 'RF_all' ,'HM_VGC', 'HM_PM'
                          ,'HM_TM','HM_VGC_PM','HM_VGC_TM','VGC_PM','VGC_TM','VGC_PM_TM','PM_TM', 'HM_PM_TM']  
    Pn_Per = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                          'SVM_add_PanelMove','SVM_add_TileMove', 'SVM_all',\
                          'RF_prior', 'RF_add_Heat', 'RF_add_cross', \
                          'RF_add_PanelMove', 'RF_add_TileMove', 'RF_all',\
                          'Y_SVM_HM_VGC','Y_SVM_HM_PM', 'Y_SVM_HM_TM', 'Y_SVM_HM_VGC_PM', 'Y_SVM_HM_VGC_TM',\
                          'Y_SVM_VGC_PM', 'Y_SVM_VGC_TM', 'Y_SVM_VGC_PM_TM', 'Y_SVM_PM_TM' ,'Y_SVM_HM_PM_TM']             
    Pg_Per = ['BL', 'SVM_prior', 'SVM_add_Heat', 'SVM_add_cross', \
                          'SVM_add_UpperPanel', 'SVM_add_UnderPanel', 'SVM_add_CorectPanel',\
                          'SVM_add_UpperTile', 'SVM_add_UnderTile', 'SVM_add_CorectTile',\
                          'SVM_PanelMat_all', 'SVM_TileMat_all', 'SVM_all',\
                          'RF_prior', 'RF_add_Heat', 'RF_add_cross', \
                          'RF_add_UpperPanel', 'RF_add_UnderPanel', 'RF_add_CorectPanel', \
                          'RF_add_UpperTile', 'RF_add_UnderTile', 'RF_add_CorectTile',\
                          'RF_PanelMat_all', 'RF_TileMat_all', 'RF_all' ,'HM_VGC', 'HM_PM'
                          ,'HM_TM','HM_VGC_PM','HM_VGC_TM','VGC_PM','VGC_TM','VGC_PM_TM','PM_TM', 'HM_PM_TM'] 
    return [Pn_Gene, Pg_Gene, Pn_Per, Pg_Per]

def Plot(alldata, i):
    cutdata = alldata.iloc[1:12,:].astype(float)
    a = pd.DataFrame()
    for column in cutdata.columns:
        for index in cutdata.index:
            a = pd.concat([a,pd.DataFrame([cutdata.loc[index, column],index,column]).T])
    a = a.rename(columns = {0:'F1', 1:'ID', 2:'Type'})
    plt.figure(figsize = (9,12))
    ax = sns.stripplot(data = a, x = 'Type', y = 'F1', hue = 'ID')
    plt.ylim(0,1)
    locs, labels = plt.xticks()
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    plt.legend(bbox_to_anchor = (1,1))
    if i == 0:
        plt.savefig('plot/GeneralPanel.png')
    elif i == 1:
        plt.savefig('plot/GeneralPage.png')
    elif i == 2:
        plt.savefig('plot/PersonalPanel.png')
    elif i == 3:
        plt.savefig('plot/PersonalPage.png')



def DataManagement(i, data_dir, nameList):
    #nameList = nameList[0]   ###for debug
    path = data_dir
    alldata = pd.DataFrame(columns = nameList)
    if i == 0:
        path += 'General/Panel/'
    elif i == 1:
        path += 'General/Page/'
    elif i == 2:
        path += 'Aggregate/Panel/'
    elif i == 3:
        path += 'Aggregate/Page/'
    for name in nameList:
        data = pd.read_csv(path + name + '.csv', encoding = 'ms932', sep = ',', index_col=0)
        alldata[name] = data.loc['f1-score',:]
    Plot(alldata, i)

def Main():
    data_dir = 'data/classification report/'
    nameList = getList()
    for i in range(len(nameList)):
        DataManagement(i, data_dir, nameList[i])
        print(nameList[i])

    
Main()