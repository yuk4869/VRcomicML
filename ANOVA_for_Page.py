# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:09:16 2022

@author: sakam
"""


import numpy as np
import pandas as pd
from scipy import stats

SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]

term = pd.read_csv('data/TextNum.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
term_Page = term.groupby('Page').sum().reset_index().drop('Panel', axis = 1)

target = 'Page'
PageColumns = [#'PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max',  
                'PanelUpperTriangul',  'PanelUnderTriangul', 'PanelCorrectOrder',
                'TileUpperTriangul',  'TileUnderTriangul', 'TileCorrectOrder',
                'FIX_Min', 'FIX_Mean','FIX_Max', 'FIX_N',
                'BLINK_D_Min', 'BLINK_D_Mean','BLINK_D_Max', 'BLINK_N',
                'GazeMove_L', 'Time', 
                'HM_Min', 'HM_Mean', 'HM_Max',  
                'VGC_Min', 'VGC_Mean', 'VGC_Max', 
                'TextNum', 'IsDifficult']
def GetData():
    EasyData = pd.DataFrame(index = PageColumns)
    DifficultData = pd.DataFrame(index = PageColumns)
    for ID in range(2,26,1):
        if ID in SkipID:continue
        described_data = pd.read_csv('data/describe/'+target+'/' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        described_data = described_data.dropna()
        described_data = pd.merge(described_data, term_Page, how = 'inner', on = ['Page']).astype('float') 
        described_data.rename(columns = {
        'DiffCrossPos_Min':'VGC_Min', 
        'DiffCrossPos_Mean':'VGC_Mean', 
        'DiffCrossPos_Max':'VGC_Max',
        'Heatmap_Min':'HM_Min',
        'Heatmap_Mean':'HM_Mean',
        'Heatmap_Max':'HM_Max'}, inplace = True)
        described_data = described_data.reindex(columns = PageColumns)
        described_data['FIX_N'] = described_data['FIX_N'] / described_data['Time']
        described_data['BLINK_N'] = described_data['BLINK_N'] / described_data['Time']
        described_data['GazeMove_L'] = described_data['GazeMove_L'] / described_data['Time']
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
    CompMean.to_csv(path_or_buf = 'data/Analize/CompareMean_Page.csv')
    print(CompMean)
    ttest = pd.DataFrame()
    for column in EasyData.columns:
        t, p = stats.ttest_ind(EasyData[column], DifficultData[column], equal_var = False)
        ttest[column] = [t,p]
    ttest.index = ['t', 'p']
    ttest.to_csv(path_or_buf = 'data/Analize/Ttest_Page.csv')
    
if __name__ == "__main__":
    Main()