# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:19:01 2021

@author: sakam
"""

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def AddHeatmapData(row_data):
    global output
    global heatmap
    global output
    
    temp = pd.DataFrame(row_data).T
    if row_data.GazePointWorldZ == 0:#マンガを見ていない場合はのちに削除できるように-1にする
        temp["HeatMap"] = -1
    else:
        Y = np.clip(int(row_data.GazePointY + 750), 0, 1500) #座標系を合わせる
        X = np.clip(int(row_data.GazePointX + 577), 0, 1155) #座標系を合わせる
        temp["HeatMap"] = heatmap.iloc[Y][X] #注視点のピクセルのヒートマップの値を取り出す
    output = output.append(temp, ignore_index = True)
    return

def MakeData(rowdata):
    global heatmap
    global page
    if page != rowdata.Page:
        page = rowdata.Page
        print(page)
        heatmap = pd.read_csv('data/HeatMap/' + str(ID) + '/' + str(page) + '.csv', encoding="ms932" ,sep=",")
    AddHeatmapData(rowdata)

scaler = MinMaxScaler([0,1])
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
output = pd.DataFrame() 
heatmap = pd.DataFrame()
page = -1
ID = -1
output = pd.DataFrame()
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]


def Add_Heatmap():
    global ID
    global output
    for ID in range(2, 26, 1):
        if ID not in SkipID:continue
        output = pd.DataFrame()
        start = time.time()
        eye_data = pd.DataFrame()
        eye_data = pd.read_csv('data/filtered/Eye' + str(ID) + '.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
        
        eye_data = eye_data.query('~IsAnnotation')
        eye_data.apply(lambda x:MakeData(x), axis = 1)
        output.to_csv(path_or_buf = 'data/AddedHeatmap/' + str(ID) + '.csv', index = False)
        print('ID' + str(ID))
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
if __name__ == "__main__":
    Add_Heatmap()