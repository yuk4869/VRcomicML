# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 07:53:01 2022

@author: sakam
"""


import numpy as np
import pandas as pd
from scipy import stats

def Data_Remake(df):
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

SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]

term = pd.read_csv('data/TextNum.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
term_Page = term.groupby('Page').sum().reset_index().drop('Panel', axis = 1)

target = 'Panel'
PanelColumns = [#'PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max', 
                'FIX_D_Min', 'FIX_D_Mean','FIX_D_Max', 'FIX_N',
                'BLINK_D_Min', 'BLINK_D_Mean','BLINK_D_Max', 'BLINK_N',
                'Panel_Move_Min','Panel_Move_Mean','Panel_Move_Max',
                'Tile_Move_Min','Tile_Move_Mean','Tile_Move_Max',
                'GazeMove_L', 'Time', 
                'HM_Min', 'HM_Mean', 'HM_Max',  
                'VGC_Min', 'VGC_Mean', 'VGC_Max', 
                'TextNum', 'IsDifficult']
def GetData():
    EasyData = pd.DataFrame(index = PanelColumns)
    DifficultData = pd.DataFrame(index = PanelColumns)
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
        described_data = Data_Remake(described_data)
        described_data = described_data.reindex(columns = PanelColumns)
        _difficultData = described_data.query('IsDifficult == 1').mean().T
        _easyData = described_data.query('IsDifficult == 0').mean().T
        _difficultData['ID'] = ID
        _easyData['ID'] = ID
        EasyData = pd.concat([EasyData, pd.DataFrame(_easyData, columns = [ID])], axis = 1)
        DifficultData = pd.concat([DifficultData, pd.DataFrame(_difficultData, columns = [ID])], axis = 1)
    return EasyData.T, DifficultData.T

def Main():
    EasyData, DifficultData = GetData()
    CompMean = pd.DataFrame( pd.concat([EasyData, DifficultData]))
    temp = pd.DataFrame(pd.concat([EasyData.mean(), DifficultData.mean()], axis = 1))
    CompMean = pd.concat([CompMean, temp.T])
    CompMean.to_csv(path_or_buf = 'data/Analize/CompareMean_Panel.csv')
    print(CompMean)
    ttest = pd.DataFrame()
    for column in EasyData.columns:
        t, p = stats.ttest_ind(EasyData[column], DifficultData[column], equal_var = False)
        ttest[column] = [t, p]
    ttest.index = ['t', 'p']
    ttest.to_csv(path_or_buf = 'data/Analize/Ttest_Panel.csv')

    
if __name__ == "__main__":
    Main()