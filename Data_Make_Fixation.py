# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:16:42 2021

@author: sakam
"""


import numpy as np
import quaternion
import pandas as pd


def Get_Eye_Pos(Head_Pos, Eye_Local_pos, Head_Rot):
    pos = []
    for i in range(len(Head_Pos[0])):
        transformed = Head_Rot[i] * Eye_Local_pos[i] * Head_Rot[i].inverse()
        pos.append([Head_Pos[0].iloc[i] + transformed.x / 1000, Head_Pos[1].iloc[i] + transformed.y / 1000, Head_Pos[2].iloc[i] + transformed.z / 1000] )
    return pd.DataFrame(np.array(pos))

def Get_Eye_Dir(Eye_Dir, Head_Rot):
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
        

def Create_Data(cut_data):
    des_data=pd.DataFrame()
    des_data['GazeMove_L'] = \
        pd.DataFrame(np.linalg.norm((cut_data.loc[:,['GazePointX','GazePointY','GazePointZ']])\
                                    .diff(),axis = 1)).dropna().sum()
    des_data['Fix_D'] = len(cut_data)
    des_data['IsDifficult'] = cut_data.IsDifficult.iloc[0]
    des_data['IsUnderstand'] = cut_data.IsUnderStand.iloc[0]
    temp = cut_data.GazePointDist - cut_data.IntersectionDist
    des_data['DiffCrossPos_Mean'] = temp.mean()
    des_data['DiffCrossPos_STD'] = temp.std()
    return des_data

def Get_past(df):
    past_panel = 0
    past_tile = 0
    if len(df.Panel.mode()) > 1:
        past_panel = df.Panel.iloc[-1]
    else : past_panel = df.Panel.mode().iloc[0]
    
    if len(df.Tile.mode()) > 1:
        past_tile = df.Tile.iloc[-1]
    else : past_tile = df.Tile.mode().iloc[0]
    return past_panel, past_tile
    
def Create_Data_Make_Intersection(cut_data, past_panel, past_tile):
    des_data=pd.DataFrame()
    GazeMove = pd.DataFrame(np.linalg.norm((cut_data.loc[:,['GazePointX','GazePointY','GazePointZ']])\
                                    .diff(),axis = 1)).dropna()
    des_data.loc[0, 'PUPIL_D_Min'] = cut_data['3Mean'].min()
    des_data['PUPIL_D_Max'] = cut_data['3Mean'].max()
    des_data['PUPIL_D_Mean'] = cut_data['3Mean'].mean()
    des_data['Panel_Move'] = (cut_data.Panel.mode() - past_panel) / cut_data.PanelNum.max()
    des_data['Tile_Move'] = (cut_data.Tile.mode() - past_tile) / 4
    des_data['GazeMove_L'] = GazeMove.sum()
    des_data['FIX_D'] = [len(cut_data)/90]
    des_data['IsDifficult'] = cut_data.IsDifficult.iloc[0]
    des_data['IsUnderstand'] = cut_data.IsUnderStand.iloc[0]
    forHeat = cut_data.query('HeatMap != -1')
    des_data['HM_Min'] = forHeat['HeatMap'].min()
    des_data['HM_Max'] = forHeat['HeatMap'].max()
    des_data['HM_Mean'] = forHeat['HeatMap'].mean()
    intersection = Intersection_Pos(cut_data)
    cut_data['IntersectionDist'] = \
        ((intersection[0] - cut_data.HeadPosX)**2 \
        + (intersection[1] - cut_data.HeadPosY)**2 \
        + (intersection[2] - cut_data.HeadPosZ)**2) ** (1/2)
    _temp = cut_data.GazePointDist - cut_data.IntersectionDist
    des_data['VGC_Min'] = _temp.min()
    des_data['VGC_Max'] = _temp.max()
    des_data['VGC_Mean'] = _temp.mean()
    des_data['Page'] = cut_data.Page.iloc[0]
    des_data['Panel'] = cut_data.Panel.mode()
    return des_data

FixationColumns = [#'PUPIL_D_Min', 'PUPIL_D_Mean', 'PUPIL_D_Max', 
                   'FIX_D', 'GazeMove_L', 
                   'HM_Min', 'HM_Mean', 'HM_Max',
                   'VGC_Min', 'VGC_Min', 'VGC_Max', 
                   'TextNum', 'IsDifficult']
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]

def Data_Make_Fixation():
    Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
    
    for ID in range(2, 26, 1):
        if ID in SkipID: continue
        described_data = pd.DataFrame()
        answer_data = pd.DataFrame()
        for i in range(1,4,1):
            answer_data = \
                pd.concat([answer_data, pd.read_csv('data/filtered/' + str(ID) + '/AnswerChapter' + str(i) + '.csv', encoding = 'ms932', sep = ',')])
    
        eye_data = pd.read_csv('data/addedHeatMap/' + str(ID) + '.csv', encoding="ms932" ,sep=",").query('Page != @Through_Page')
        
        for Page in range(2,76,1):
            queried = eye_data.query('Page == @Page')
            queried['PanelNum'] = queried.Panel.max()
            past_panel = 0
            past_tile = 0
            for j, fix in queried.groupby([(queried.Fixation != queried.Fixation.shift()).cumsum()]):
                if sum(fix.IsAnnotation) == 0 and fix.Fixation.iloc[0] == 1 and fix.Panel.iloc[0] != -1:
                    temp = fix.drop('Fixation', axis = 1).drop('Saccade', axis = 1).astype('float')
                    join_data = pd.merge(temp, answer_data, how = 'left', on = ['Page','Panel'])
                    if len(join_data) > 0:
                        described_data = pd.concat([described_data,Create_Data_Make_Intersection(join_data, past_panel, past_tile)], axis = 0)
                        past_panel, past_tile = Get_past(fix)
        described_data.to_csv(path_or_buf = 'data/describe/Fixation/' + str(ID) + '.csv',index = False)
        print(ID)
        
if __name__ == "__main__":
    Data_Make_Fixation()