# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:47:42 2021

@author: sakam
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Data_Remake(df):
    remade = pd.DataFrame(columns = df.columns)
    for page, panel in zip(answer_data['Page'], answer_data['Panel']):
        new = pd.DataFrame(columns = df.columns)
        queried = df.query('Page == @page & Panel == @panel')
        if len(queried) < 1:continue
        new.loc[0, 'Page'] = page
        new['Panel'] = panel
        new['IsDifficult']  = queried['IsDifficult'].max()
        new['IsUnderstand'] = queried['IsUnderStand'].max()
        remade = remade.append(new)
    return remade    

def stacked_graph(dataset):  
    plt.rcParams["font.size"] = 30
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(dataset)):
        ax.bar(dataset.columns, dataset.iloc[i], bottom=dataset.iloc[:i].sum())
    plt.title(target)
    plt.xticks(np.arange(min(dataset.columns), max(dataset.columns)+1, 1.0))
    ax.set(xlabel='Participants id', ylabel='The number of data')
    ax.legend(dataset.index, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.savefig("plot/" + target + "Stack.png",bbox_inches="tight")
    plt.show()



targets = ['Fixation', 'Panel', 'Page']
Through_Page = [1,5,19,20,21,29,31,36,43,44,45,46,47,60,61,62,63,74,75,76]
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]

answer_data = pd.DataFrame()
for i in range(1,4,1):
    answer_data = \
        pd.concat([answer_data, pd.read_csv('data/filtered/18/AnswerChapter' + str(i) + '.csv', encoding = 'ms932', sep = ',')])
answer_data = answer_data.query('Page != @Through_Page')

for target in targets:
    data = pd.DataFrame()
    i = 0
    for ID in range(2,26,1):
        if ID in SkipID:continue
        i += 1
        eyedata = pd.read_csv('data/describe/' + target +'/' + str(ID) + '.csv', encoding="ms932", sep=",").dropna()
        if target == 'Panel':eyedata = Data_Remake(eyedata)
        data[i] = eyedata['IsDifficult'].astype('int').value_counts()
    data = data.reindex(index=[0, 1])
    data = data.rename(index={0: 'Easy', 1: 'Difficult'})
    stacked_graph(data.astype('int'))

j = 0
data = pd.DataFrame()
for ID in range(2,26,1):
    j+=1
    answer_data = pd.DataFrame()
    for i in range(1,4,1):
        answer_data = \
            pd.concat([answer_data, pd.read_csv('data/raw/'+str(ID)+'/AnswerChapter' + str(i) + '.csv', encoding = 'ms932', sep = ',')])
    answer_data = answer_data.query('Page != @Through_Page')
    data[j] = answer_data['IsDifficult'].astype('int').value_counts()
data = data.reindex(index=[0, 1])
data = data.rename(index={0: 'Easy', 1: 'Difficult'})
stacked_graph(data.astype('int'))

