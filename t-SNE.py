# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:52:57 2022

@author: sakam
"""
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
import itertools
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

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
                'TextNum', 'IsDifficult', "ID"]
def GetData():
    Data = pd.DataFrame(columns = PanelColumns)
    for ID in tqdm(range(2,26,1)):
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
        described_data["ID"] = ID
        described_data = described_data.reindex(columns = PanelColumns)
        Data = pd.concat([Data, pd.DataFrame(described_data, columns = PanelColumns)])
    return Data

def PlotEach(Data, columns, dimension):
    drop_columns = ['TextNum', 'ID', 'IsDifficult']
    tsne = TSNE(n_components=dimension, random_state=912, method = 'exact')
    for ID in tqdm(range(24,26,1)):
        if ID in SkipID:continue
        query = Data.query("ID == @ID").drop(drop_columns, axis = 1)
        X_reduced = tsne.fit_transform(query)
        if dimension == 2:
            Plot(X_reduced, Data.query("ID == @ID")["IsDifficult"], '_'.join(columns) + " " + str(ID))
        elif dimension == 3:
            Plot3d(X_reduced, Data.query("ID == @ID")["IsDifficult"], '_'.join(columns) + " " + str(ID))

def PlotAll(Data, columns, dimension):
    drop_columns = ['TextNum', 'ID', 'IsDifficult']
    tsne = TSNE(n_components=dimension, random_state=912, method = 'exact')
    X = Data.drop(drop_columns, axis = 1)
    X_reduced = tsne.fit_transform(X)
    if dimension == 2:
        Plot(X_reduced, Data["IsDifficult"], '_'.join(columns) + " all")
    elif dimension == 3:
        Plot3d(X_reduced, Data["IsDifficult"], '_'.join(columns) + " all")
        
def Plot(X_reduced, IsDifficult, title):
    plt.figure(figsize=(13, 7))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1],\
            c=IsDifficult.map({1:'b', 0:'g'}), s=15, alpha = 0.5)
    plt.axis('off')
    plt.title(title)
    plt.colorbar()
    plt.savefig("plot/t-sne/panel/2d/exact/" + title)
    plt.show()

def Plot3d(X_reduced, IsDifficult, title):
    print(X_reduced.size)
    plt.figure(figsize=(13, 7)).gca(projection='3d')
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=IsDifficult.map({1:'b', 0:'g'}))
    plt.title(title)
    plt.colorbar()
    plt.savefig("plot/t-sne/panel/3d/" + title)
    plt.show()

def Main():
    Data = GetData()
    columns = ['VGC', 'HM', 'Tile', 'Panel']
    columns_list = np.logical_not(Data.columns.str.contains('|'.join(columns)))
    PlotAll(Data.loc[:,columns_list], ["prior"], 2)
    PlotEach(Data.loc[:,columns_list], ["prior"], 2)
    for i in tqdm(range(1,5,1)):
        for picked_columns in itertools.combinations(columns, i):
            columns_list = np.logical_not(Data.columns.str.contains('|'.join(columns)))
            columns_list += Data.columns.str.contains('|'.join(picked_columns))
            #Plot_tsne(Data.loc[:,columns_list], picked_columns)
            PlotAll(Data.loc[:,columns_list], picked_columns, 2)
            PlotEach(Data.loc[:,columns_list], picked_columns, 2)
            
            
if __name__ == "__main__":
    Main()
    
    