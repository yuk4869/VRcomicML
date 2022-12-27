# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:08:57 2021

@author: sakam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_sacfix(df):
    fig, ax1 = plt.subplots()
    x = range(len(df))
    xlim = [1400,1500]
    
    ax1.plot(x, df['X'],  linewidth=4, color="blue")
    #ax1.set_ylabel('left_x')
    ax1.set_xlim(xlim)
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    ax2.plot(x, df['Y'], linewidth=4, color="orange")
    #ax2.set_ylabel('left_y')
    ax2.set_xlim(xlim)
    ax2.set_ylim(0, 1)
    
    ax3 = ax1.twinx()
    ax3.plot(x, df['Fixation'], linewidth=1, color="green")
    #ax3.set_ylabel('isFixation')
    ax3.set_xlim(xlim)
    ax3.set_ylim(-0.5,2)

class Detector(object): 
    @staticmethod
    def detectSaccade(inputArrayX = [], inputArrayY = [], *args):
        fixationThreshold = 0.00015 #maximum distance between points for fixation
        if math.sqrt((inputArrayX[2]-inputArrayX[3])**2+\
                     (inputArrayY[2]-inputArrayY[3])**2) > fixationThreshold \
        and math.sqrt((inputArrayX[1]-inputArrayX[2])**2+\
                     (inputArrayY[1]-inputArrayY[2])**2) < fixationThreshold:
            return True
        else:
            return False
    @staticmethod
    def detectFixation(inputArrayX = [], inputArrayY = [], *args):
        saccadeThreshold = 0.0025 #minimum distance between points for saccade
        if math.sqrt((inputArrayX[0]-inputArrayX[1])**2+\
                     (inputArrayY[0]-inputArrayY[1])**2) < saccadeThreshold \
        and math.sqrt((inputArrayX[1]-inputArrayX[2])**2+\
                     (inputArrayY[1]-inputArrayY[2])**2) < saccadeThreshold \
        and math.sqrt((inputArrayX[2]-inputArrayX[3])**2+\
                     (inputArrayY[2]-inputArrayY[3])**2) < saccadeThreshold :

            return True
        else:
            return False

#極端に近いFixationを合成する
def Merge_Fixation(result):
    FirstFlag = True
    _result = pd.DataFrame()
    Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
    for Page in range(1,77, 1):
        if Page in Through_Page:continue
        queried = result.query('Page == @Page')
        for j, fix in queried.groupby([(queried.Fixation != queried.Fixation.shift()).cumsum()]):
            if FirstFlag:
                FirstFlag = False
                _result = _result.append(fix)
                continue
            if sum(fix.Fixation) == 0 and len(fix) < BETWEEN_FIXATION:
                fix.Fixation = 1
            _result = pd.concat([_result, fix])
    return _result

#極端に近いFixationを削除
def Discard_Fixation(result):
    FirstFlag = True
    _result = pd.DataFrame()    
    Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
    for Page in range(1,77, 1):
        if Page in Through_Page:continue
        queried = result.query('Page == @Page')
        for j, fix in queried.groupby([(queried.Fixation != queried.Fixation.shift()).cumsum()]):
            if FirstFlag:
                FirstFlag = False
                _result = _result.append(fix)
                continue
            if sum(fix.Fixation) != 0 and len(fix) < MAX_FIXATION_DURATION:
                fix.Fixation = 0
            _result = pd.concat([_result, fix])
    return _result

#瞬きの間もFixation判定されてしまう場合があるため削除
def Discard_Fixation_In_Blink(result):
    FirstFlag = True
    _result = pd.DataFrame()    
    Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
    for Page in range(1,77, 1):
        if Page in Through_Page:continue
        queried = result.query('Page == @Page')
        for j, blink in queried.groupby([(queried.IsBlink != queried.IsBlink.shift()).cumsum()]):
            if FirstFlag:
                FirstFlag = False
                _result = _result.append(blink)
                continue
            if sum(blink.IsBlink) != 0:
                blink.Fixation = 0
            _result = pd.concat([_result, blink])
    return _result

BETWEEN_FIXATION = 7 #7frame ≒ 77.7ms
MAX_FIXATION_DURATION = 6 #6frame ≒ 66.6ms
numFrames = 4 #numFrameの間の移動が一定以下ならFixation
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]
test_len = 1500
for ID in range(2, 26, 1):
    if ID in SkipID: continue
    eye_data = pd.read_csv('data/filtered/Eye' + str(ID) + '.csv', encoding = 'ms932', sep = ',')
    framesX = [] #holds an queue of frames (x column) 
    framesY = [] #holds an queue of frames (y column)
    eye_data['MedLeftX'] = eye_data['LeftDisplaceNormalizedX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['MedLeftY'] = eye_data['LeftDisplaceNormalizedY'].rolling(3,min_periods = 1, center = True).median()
    
    
    result = pd.DataFrame(columns = {'X', 'Y', 'Fixation'})
    for row in range(len(eye_data)):
        #if row == test_len: #limit rows to 50 for debugging
        #    break #for debugging
        isFixation = 0
        if len(framesX) < numFrames: #append data to frame queues
            framesX.append(float(eye_data.loc[row].MedLeftX))
            framesY.append(float(eye_data.loc[row].MedLeftY))
            
        if len(framesX) >= numFrames and Detector.detectFixation(framesX, framesY): #judge
            isFixation = 1
            
        if len(framesX) >= numFrames:
            framesX.pop(0)
            framesY.pop(0)
        result = result.append({'X': float(eye_data.loc[row].MedLeftX),\
                                'Y': float(eye_data.loc[row].MedLeftY), \
                                'Fixation': isFixation\
                                }, ignore_index=True)
    result['Page'] = eye_data['Page']
    result['IsBlink'] = eye_data['IsBlink']
    result = Merge_Fixation(result)
    result = Discard_Fixation(result)
    result = Discard_Fixation_In_Blink(result)
    eye_data['Fixation'] = result['Fixation']
    eye_data.to_csv(path_or_buf = 'data/filtered/Eye' + str(ID) + '.csv',index = False)
    print(str(ID))
        
        
##result check
resultList = []
for j, fix in result.groupby([(result.Fixation != result.Fixation.shift()).cumsum()]):
    if (fix.Fixation.iloc[0]):
        resultList.append(len(fix))
pd.DataFrame(resultList).describe()
plot_sacfix(result)

def remake():    
    for ID in range(2, 19, 1):
        if ID == 6 or ID == 14 or ID == 15: continue
        eye_data = pd.read_csv('data/Eye' + str(ID) + '.csv', encoding = 'ms932', sep = ',')
        result = Discard_Fixation_In_Blink(eye_data)
        eye_data['Fixation'] = result['Fixation']
        result.to_csv(path_or_buf = 'data/filtered/Eye' + str(ID) + '.csv',index = False)
        print(ID)
        