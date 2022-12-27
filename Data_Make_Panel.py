# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:24:56 2021

@author: sakam
"""

import numpy as np
import pandas as pd
import quaternion
import time

def Get_Eye_Pos(Head_Pos, Eye_Local_pos, Head_Rot): #GazeOriginはHMDの位置に対するlocal positionだからワールド座標に変換
    pos = []
    for i in range(len(Head_Pos[0])):
        transformed = Head_Rot[i] * Eye_Local_pos[i] * Head_Rot[i].inverse()
        pos.append([Head_Pos[0].iloc[i] + transformed.x / 1000, Head_Pos[1].iloc[i] + transformed.y / 1000, Head_Pos[2].iloc[i] + transformed.z / 1000] )
    return pd.DataFrame(np.array(pos))

def Get_Eye_Dir(Eye_Dir, Head_Rot): #GazeDirectionもワールド座標に変換
    direction = []
    for i in range(len(Head_Rot)):
        transformed = Head_Rot[i] * Eye_Dir[i] * Head_Rot[i].inverse()
        direction.append([transformed.x, transformed.y, transformed.z])
    return pd.DataFrame(np.array(direction))

def Intersection_Pos(cut_data):
    cut_data.index = range(len(cut_data))
    LeftOriginArray = np.array([[0] * len(cut_data), -cut_data.LeftOriginPosX, cut_data.LeftOriginPosY, cut_data.LeftOriginPosZ]).T
    RightOriginArray = np.array([[0] * len(cut_data), -cut_data.RightOriginPosX, cut_data.RightOriginPosY, cut_data.RightOriginPosZ]).T
    LeftDirArray = np.array([[0] * len(cut_data), -cut_data.LeftGazeDirectionX, cut_data.LeftGazeDirectionY, cut_data.LeftGazeDirectionZ]).T
    RightDirArray = np.array([[0] * len(cut_data), -cut_data.RightGazeDirectionX, cut_data.RightGazeDirectionY, cut_data.RightGazeDirectionZ]).T
    HeadArray = np.array([cut_data.HeadRotationW, cut_data.HeadRotationX, cut_data.HeadRotationY,cut_data.HeadRotationZ]).T
    
    LeftPos =  Get_Eye_Pos( \
        [cut_data.HeadPosX, cut_data.HeadPosY,cut_data.HeadPosZ], \
        quaternion.from_float_array(LeftOriginArray), \
        quaternion.from_float_array(HeadArray))
    RightPos = Get_Eye_Pos( \
        [cut_data.HeadPosX, cut_data.HeadPosY,cut_data.HeadPosZ], \
        quaternion.from_float_array(RightOriginArray), \
        quaternion.from_float_array(HeadArray))
    LeftDir = Get_Eye_Dir( \
        quaternion.from_float_array(LeftDirArray), \
        quaternion.from_float_array(HeadArray))
    RightDir = Get_Eye_Dir( \
        quaternion.from_float_array(RightDirArray), \
        quaternion.from_float_array(HeadArray))
        
    dot = LeftDir[0] * RightDir[0] \
        + LeftDir[1] * RightDir[1] \
        + LeftDir[2] * RightDir[2]
    
    work = 1 - dot * dot

    Between_Eye_VecX = RightPos[0] - LeftPos[0]
    Between_Eye_VecY = RightPos[1] - LeftPos[1]
    Between_Eye_VecZ = RightPos[2] - LeftPos[2]
    
    d1  = Between_Eye_VecX * LeftDir[0] \
        + Between_Eye_VecY * LeftDir[1] \
        + Between_Eye_VecZ * LeftDir[2]
    d2  = Between_Eye_VecX * RightDir[0] \
        + Between_Eye_VecY * RightDir[1] \
        + Between_Eye_VecZ * RightDir[2]
    d = (d1 - dot * d2 ) / work 
    d3 = (d1 * dot - d2) / work
    return [ \
        LeftPos[0] + d * LeftDir[0], \
        LeftPos[1] + d * LeftDir[1], \
        LeftPos[2] + d * LeftDir[2],  \
        RightPos[0] + d3 * RightDir[0],\
        RightPos[1] + d3 * RightDir[1],\
        RightPos[2] + d3 * RightDir[2]
    ]
        
def count_blink(cut_data):
    lcb = cut_data['IsBlink']
    blink_count = lcb.groupby(lcb.cumsum()).count()
    if blink_count[blink_count != 1].empty: #瞬きしていないときはnoneになるので回避用
        count = pd.DataFrame({'BLINK_N':0,
                              'BLINK_D_Min':0,
                              'BLINK_D_Max':0,
                              'BLINK_D_Mean':0,
                              'BLINK_D_Sum':0,
                              'BLINK_D_RATE':0},
                             index = [0])
    else :
        describe = blink_count[blink_count != 1].describe()
        count = pd.DataFrame({'BLINK_N':describe['count'],
                         'BLINK_D_Min':describe['min'] / 90,
                         'BLINK_D_Max':describe['max'] / 90,
                         'BLINK_D_Mean':describe.loc['mean'] / 90,
                         'BLINK_D_Sum':blink_count[blink_count != 1].sum() / 90,
                         'BLINK_D_RATE':blink_count[blink_count != 1].sum() / len(cut_data) 
                         },
                         index = [0])
    count = count.fillna(0)
    return count

def count_fix(cut_data):
    count = pd.DataFrame()
    fix_df = pd.DataFrame(cut_data['Fixation'])
    fix_df['FixCount'] = fix_df.groupby((fix_df.Fixation != fix_df.Fixation.shift()).cumsum()).cumcount() + 1 
    des = fix_df.groupby((fix_df.Fixation==0).cumsum()).count().query('Fixation != 1').describe()
    count.loc[0, 'FIX_N']= des.loc['count','Fixation']
    count.loc[0, 'FIX_Min'] = des.loc['min','Fixation'] / 90
    count.loc[0, 'FIX_Max'] = des.loc['max','Fixation'] / 90
    count.loc[0, 'FIX_Mean'] = des.loc['mean','Fixation'] / 90
    count.loc[0, 'FIX_Sum'] = fix_df.groupby((fix_df.Fixation==0).cumsum()).count().query('Fixation != 1').sum().Fixation / 90
    count['FIX_RATE'] = count['FIX_Sum'] * 90 / len(cut_data)
    return count

def Create_Data(cut_data, past_panel, past_tile):
    des_data=pd.DataFrame()
    des_data = pd.concat([des_data,count_blink(cut_data)],axis=1)
    des_data = pd.concat([des_data,count_fix(cut_data)],axis=1)
    filterd = cut_data[(cut_data['IsBlink'] == False)& \
                  (cut_data['GazePointWorldZ'] != 0)]
    des_data['FrameCount'] = len(cut_data)
    des_data['Time'] = len(cut_data)/90
    des_data['PUPIL_D_Min'] = filterd['3Mean'].min()
    des_data['PUPIL_D_Max'] = filterd['3Mean'].max()
    des_data['PUPIL_D_Mean'] = filterd['3Mean'].mean()
    des_data['Panel_Move'] = (cut_data.Panel.max() - past_panel) / cut_data.PanelNum.max()
    des_data['Tile_Move'] = (cut_data.Tile.mean() - past_tile) / 4
    des_data['GazeMove_L'] = \
        pd.DataFrame(np.linalg.norm((cut_data.loc[:,['GazePointX','GazePointY','GazePointZ']])\
                                    .diff(),axis = 1)).dropna().sum()
    forHeat = filterd.query('HeatMap != -1').query('Fixation == 1')
    des_data['HM_Min'] = forHeat['HeatMap'].min()
    des_data['HM_Max'] = forHeat['HeatMap'].max()
    des_data['HM_Mean'] = forHeat['HeatMap'].mean()
    intersection = Intersection_Pos(filterd)
    filterd['IntersectionDist'] = \
        ((-intersection[0] - filterd.HeadPosX)**2 \
        + (intersection[1] - filterd.HeadPosY)**2 \
        + (intersection[2] - filterd.HeadPosZ)**2) ** (1/2)
    temp = filterd.GazePointDist - filterd.IntersectionDist
    des_data['VGC_Min'] = temp.min()
    des_data['VGC_Max'] = temp.max()
    des_data['VGC_Mean'] = temp.mean()
    des_data['Page'] = cut_data.Page.iloc[0]
    des_data['Panel'] = cut_data.Panel.iloc[0]
    return des_data

PanelColumns = ['PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max', 
                'FIX_D_Min', 'FIX_D_Mean','FIX_D_Max', 'FIX_N',
                'BLINK_D_Min', 'BLINK_D_Mean','BLINK_D_Max', 'BLINK_N',
                'GazeMove_L', 'Time', 
                'HM_Min', 'HM_Mean', 'HM_Max',  
                'VGC_Min', 'VGC_Mean', 'VGC_Max',
                'TextNum', 'IsDifficult']
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]


def Data_Make_Panel():
    Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
    
    for ID in range(2,26, 1):
        if ID in SkipID: continue
        start = time.time()
        described_list_panel = pd.DataFrame()
        answer_data = pd.DataFrame()
        for i in range(1,4,1):
            answer_data = \
                pd.concat([answer_data, pd.read_csv('data/filtered/' + str(ID) + '/AnswerChapter' + str(i) + '.csv', encoding = 'ms932', sep = ',')])
    
        eye_data = pd.read_csv('data/AddedHeatmap/' + str(ID) + '.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
        
        for Page in range(2,76,1):
            queried = eye_data.query('Page == @Page')
            queried['PanelNum'] = queried.Panel.max()
            past_panel = 0
            past_tile = 0
            for j, data in queried.groupby([(queried.Panel != queried.Panel.shift()).cumsum()]):
                filterd = data[(data['IsBlink'] == False)& \
                                       (data['GazePointWorldZ'] != 0)]
                if len(filterd) < 1:
                    continue
                described_list_panel = pd.concat([described_list_panel, Create_Data(data, past_panel, past_tile)]\
                                                 ,ignore_index = True) 
                past_panel = data.Panel.max()
                past_tile = data.Tile.mean()
                
        panel_data = pd.merge(described_list_panel, answer_data, how = 'left', on = ['Page','Panel'])
        panel_data.to_csv(path_or_buf = 'data/describe/Panel/' + str(ID) + '.csv', index = False)
        print(ID)
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        
        
if __name__ == "__main__":
    Data_Make_Panel()