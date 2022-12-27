# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:54:09 2022

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
                'TextNum', 'IsDifficult', "ID"]
def GetData():
    Data = pd.DataFrame(columns = PageColumns)
    for ID in tqdm(range(2,26,1)):
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
        described_data["ID"] = ID
        described_data = described_data.reindex(columns = PageColumns)
        Data = pd.concat([Data, pd.DataFrame(described_data, columns = PageColumns)])
    return Data

def PlotEach(Data, columns, dimension):
    drop_columns = ['TextNum', 'ID', 'IsDifficult']
    tsne = TSNE(n_components=dimension, random_state=912, method = 'exact')
    for ID in tqdm(range(24,26,1)):
        if ID in SkipID:continue
        query = Data.query("ID == @ID").drop(drop_columns, axis = 1)
        X_reduced = tsne.fit_transform(query)
        Plot(X_reduced, Data.query("ID == @ID")["IsDifficult"], '_'.join(columns) + " " + str(ID), dimension )

def PlotAll(Data, columns, dimension):
    drop_columns = ['TextNum', 'ID', 'IsDifficult']
    tsne = TSNE(n_components=dimension, random_state=912, method = 'exact')
    X = Data.drop(drop_columns, axis = 1)
    X_reduced = tsne.fit_transform(X)
    Plot(X_reduced, Data["IsDifficult"], '_'.join(columns) + " all", dimension)

        
def Plot(X_reduced, IsDifficult, title, dimension):
    if dimension == 2:
        Plot2d(X_reduced, IsDifficult, title)
    elif dimension == 3:
        Plot3d(X_reduced, IsDifficult, title)
 

def Plot2d(X_reduced, IsDifficult, title):
    plt.figure(figsize=(13, 7))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1],\
            c=IsDifficult.map({1:'b', 0:'g'}), s=15, alpha = 0.5)
    plt.axis('off')
    plt.title(title)
    plt.colorbar()
    plt.savefig("plot/t-sne/page/2d/exact/" + title)
    plt.show()

def Plot3d(X_reduced, IsDifficult, title):
    print(X_reduced.size)
    plt.figure(figsize=(13, 7)).gca(projection='3d')
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=IsDifficult.map({1:'b', 0:'g'}))
    plt.title(title)
    plt.colorbar()
    plt.savefig("plot/t-sne/page/3d/" + title)
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
            PlotAll(Data.loc[:,columns_list], picked_columns, 2)
            PlotEach(Data.loc[:,columns_list], picked_columns, 2)
            
            
if __name__ == "__main__":
    Main()
    
    