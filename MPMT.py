# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:33:09 2022

@author: sakam
"""


import pandas as pd
import matplotlib.pyplot as plt

Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
def Plot(df):
    df = df.reset_index()
    #plt.title("Movement between panels or tiles")
    Page = df.Page[0]
    plt.title(str(Page) +":" + str(answer_data.query("Page == @Page").IsDifficult.sum()))
    plt.plot(df.Tile, color = "red", label = "Tile")
    plt.plot(df.Panel, color = "blue", label = "Panel")
    plt.xlim(100,len(df)-1)
    plt.ylim(0.5,4.5)
    plt.subplot().yaxis.set_ticks(pd.np.arange(1,5,1))
    plt.xlabel("Number of frame")
    plt.ylabel("Panel or tile number")
    plt.legend(bbox_to_anchor = [1,1])
    plt.show()
    
answer_data = pd.DataFrame()

for i in range(1,4,1):
    answer_data = \
        pd.concat([answer_data, pd.read_csv('data/filtered/3/AnswerChapter' + str(i) + '.csv', encoding = 'ms932', sep = ',')])

data = pd.read_csv("data/AddedHeatmap/3.csv")
for Page in range(1,77,1):
    if Page in Through_Page:
        continue
    Plot(data.query("Page == @Page"))

data = pd.read_csv("data/raw/24/Chapter3.csv").query("Page == 57")
plt.plot(data.GazePointX, color = "red")
plt.plot(data.GazePointY, color = "blue")

