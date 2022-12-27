# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:20:20 2020

@author: sakam
"""

import matplotlib.pyplot as plt
import pandas as pd
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]


#3フレームの移動中央値を使ってフィルタリング
#平均だと平滑化されすぎてFixation判定に影響が出てしまう
#中央値にすることでスパイクノイズだけを除去
for ID in range(2, 6, 1):  #26 to 6
    eye_data = pd.DataFrame()
    for i in range(1,4,1):
        temp = pd.read_csv('data/raw/' + str(ID) + '/Chapter' + str(i) + '.csv', encoding="ms932" ,sep=",")
        eye_data = pd.concat([eye_data, temp])
    eye_data['BothAverage'] = (eye_data['LeftDiameter'] + eye_data['RightDiameter']) /2
    eye_data['3Mean'] = eye_data['BothAverage'].rolling(3,min_periods = 1, center = True).mean()
    eye_data['3Med'] = eye_data['BothAverage'].rolling(3,min_periods = 1, center = True).median()
    eye_data['D_vel'] = eye_data['3Med'].diff()
    eye_data['Blink'] = (0.49 < eye_data['D_vel'].abs()) | (eye_data['3Med'] < 1.86)
    eye_data['LeftOpenness'] = eye_data['LeftOpenness'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightOpenness'] = eye_data['RightOpenness'].rolling(3,min_periods = 1, center = True).median()
    eye_data['IsBlink'] = eye_data['Blink'] | (eye_data['LeftOpenness'] < 0.5) | (eye_data['RightOpenness'] < 0.5)
    eye_data['GazePointX'] = eye_data['GazePointX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['GazePointY'] = eye_data['GazePointY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadPosX'] = eye_data['HeadPosX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadPosY'] = eye_data['HeadPosY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadPosZ'] = eye_data['HeadPosZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadRotationX'] = eye_data['HeadRotationX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadRotationY'] = eye_data['HeadRotationY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['HeadRotationZ'] = eye_data['HeadRotationZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftOriginPosX'] = eye_data['LeftOriginPosX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftOriginPosY'] = eye_data['LeftOriginPosY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftOriginPosZ'] = eye_data['LeftOriginPosZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightOriginPosX'] = eye_data['RightOriginPosX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightOriginPosY'] = eye_data['RightOriginPosY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightOriginPosZ'] = eye_data['RightOriginPosZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftGazeDirectionX'] = eye_data['LeftGazeDirectionX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftGazeDirectionY'] = eye_data['LeftGazeDirectionY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['LeftGazeDirectionZ'] = eye_data['LeftGazeDirectionZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightGazeDirectionX'] = eye_data['RightGazeDirectionX'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightGazeDirectionY'] = eye_data['RightGazeDirectionY'].rolling(3,min_periods = 1, center = True).median()
    eye_data['RightGazeDirectionZ'] = eye_data['RightGazeDirectionZ'].rolling(3,min_periods = 1, center = True).median()
    eye_data.to_csv(path_or_buf ='data/filtered/Eye' + str(ID) + '.csv', index = False)


####for check result####
eye_data['D_vel'] = eye_data['3Med'].diff()
eye_data['Blink'] = (0.49 < eye_data['D_vel'].abs()) | (eye_data['3Med'] < 1.86)
eye_data['IsBlink'] = eye_data['Blink'] | (eye_data['LeftOpenness'] < 0.5) | (eye_data['RightOpenness'] < 0.5)
eye_data['OnlyThreshold'] = (eye_data['LeftOpenness'] < 0.9) | (eye_data['RightOpenness'] < 0.9)
eye_data['AroundBlink'] = (eye_data['3Med']/eye_data['3Mean']>1.1) | (eye_data['3Med']/eye_data['3Mean']<0.9)


def plot_result(df, x_min, x_max):
    fig, ax1 = plt.subplots()
    x = range(len(df))
    xlim = [x_min, x_max]
    
    ax1.plot(x, df['OnlyThreshold'], linewidth=2, color="orange", label = 'old algorithm')
    ax1.plot(x, df['IsBlink'],  linewidth=2, color="blue", label = 'new algorithm')
    ax1.plot(x, df['LeftOpenness'], linewidth=4, color="black", label = 'Openness')    
    ax1.set_xlim(xlim)
    ax1.set_ylim(-1, 2)
    plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
    
    ax3 = ax1.twinx()
    ax3.plot(x, df['BothAverage'], linewidth=1, color="black", label = 'raw')
    ax3.plot(x, df['3Mean'], linewidth=1, color="green", label = '3 frame mean')
    ax3.plot(x, df['3Med'], linewidth=1, color="pink", label = '3 frame med')
    ax3.set_xlim(xlim)
    ax3.set_ylim(2,6)
    plt.legend(bbox_to_anchor=(1.1, 0.7), loc='upper left', borderaxespad=0)
