# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:11:04 2022

@author: sakam
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm

initH = 1500 #マンガの縦の解像度
initW = 1155 #マンガの横の解像度
size = 1 #ヒートマップの解像度の比　１でマンガと同サイズ、100なら1/100倍
output = pd.DataFrame()

class FDM:
    def __init__(self, H, W):
        self.H_steps = np.linspace(0., 1., H, endpoint=False)
        self.W_steps = np.linspace(0., 1., W, endpoint=False)
        self.H = H
        self.W = W
        self.Size = np.dstack(np.meshgrid(self.W_steps, self.H_steps)).reshape((-1, 2))
        self.Map = np.array([[0] * W for i in range(H)], dtype = float)
    
    def make_heatmap(self, gaze_point_y, gaze_point_x):
        gp = np.vstack((np.array(gaze_point_x), np.array(gaze_point_y))).astype(np.float64)
        kde = sm.nonparametric.KDEMultivariate(data=[gp[0,:], gp[1,:]], var_type='cc', bw=[0.05, 0.05])
        self.Map = (kde.pdf(self.Size) * gp.shape[1]).reshape((self.H, self.W))


def main():
    SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]
    for ID in range(2, 26, 1):
        if ID in SkipID: continue
        eye_data = pd.read_csv('data/filtered/Eye' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        onlyFix = eye_data.query('Fixation == 1')
        hList = (onlyFix.GazePointY + (initH/2)) / initH
        wList = (-onlyFix.GazePointX + (initW/2)) / initW
        H = (int)(initH / size)
        W = (int)(initW / size)
        fdm = FDM(H, W)
        fdm.make_heatmap(hList, wList)
        pd.DataFrame(fdm.Map).to_csv(path_or_buf = 'data/AddedHeatmap/' + str(ID) + '.csv', index = False)

if __name__ == '__main__':
    main()