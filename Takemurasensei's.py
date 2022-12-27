# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cv2
import os


#ヒートマップ配列の作成（STEPS x STEPS）
STEPS=100
s = np.linspace(0., 1., STEPS, endpoint=False)
s = np.dstack(np.meshgrid(s, s)).reshape((-1, 2))


#GazePointの読み込み
gp = pd.read_csv("./sample.csv")
x = np.vstack((np.array(gp['GazePoint_X']), np.array(gp['GazePoint_Y']))).astype(np.float64)


#ヒートマップ作成（カーネル密度推定）
kde = sm.nonparametric.KDEMultivariate(data=[x[0,:], x[1,:]], var_type='cc', bw=[0.05, 0.05])
heatmap=(kde.pdf(s) * x.shape[1]).reshape((STEPS, STEPS))


#描画（おまけ）
plt.imshow(heatmap, vmin=0, vmax=800)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(bottom=False, left=False, right=False, top=False)
plt.savefig('heatmap.png',  bbox_inches='tight', pad_inches=0.0)
plt.close()
