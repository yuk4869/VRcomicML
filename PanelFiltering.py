# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:07:52 2021

@author: sakam
"""


import pandas as pd

PastPage = -1
PastPanel = -1
ResetedPanel = pd.DataFrame()
MIN_BLINK_DURATION = 7

def ResetPanel(row): 
    #瞬きの間やノイズ等でマンガ以外を見ていると判定されてしまうことがあるため、見ているコマをフィルタリング
        #連続で6フレーム以上見ている→異なるコマに移動
        #6フレーム未満→直前に見ていたコマに滞在中
    
    global ResetedPanel
    global PastPanel
    global PastPage
    
    if PastPage != row[1]:
        PastPanel = -1
        PastPage = row[1]
    
    if (row[2]) or (row[0] == 0) or (row[0] == -1):
        ResetedPanel = ResetedPanel.append([PastPanel])
    else:
        ResetedPanel = ResetedPanel.append([row[0]])
        PastPanel = row[0]

def Discard_Blink(df):
    FirstFlag = True
    _result = pd.DataFrame()
    for j, blink in df.groupby([(df.IsBlink != df.IsBlink.shift()).cumsum()]):
        if FirstFlag:
            FirstFlag = False
            _result = _result.append(blink)
            continue
        if sum(blink.IsBlink) < MIN_BLINK_DURATION:
            blink.IsBlink = 0
        _result = pd.concat([_result, blink])
    return _result

def Discard_short_look(Panel_Data):
    global ResetedPanel
    _pastPanel = -1
    filterdPanel = pd.DataFrame()
    
    ResetedPanel = ResetedPanel.reset_index(drop = True) 
    ResetedPanel.Page = Panel_Data['Page']
    for j, page in ResetedPanel.groupby([(ResetedPanel.Page != ResetedPanel.Page.shift()).cumsum()]):
        _pastPanel = -1
        for k, panel in page.groupby([(page[0] != page[0].shift()).cumsum()]):
            if len(panel) < MIN_LOOK:
                panel[0] = _pastPanel
            filterdPanel = pd.concat([filterdPanel, panel[0]])
            _pastPanel = panel.iloc[0,0]
    return filterdPanel

MIN_LOOK = 6
SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]

def main():
    global ResetedPanel
    global PastPanel
    global PastPage
    
    for ID in range(2, 26, 1):
        if ID in SkipID: continue
        eye_data = pd.read_csv('data/AddedHeatmap/' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        eye_data['IsBlink'] = eye_data['Blink'] | (eye_data['LeftOpenness'] < 0.5) | (eye_data['RightOpenness'] < 0.5)
        Panel_Data = pd.DataFrame()
        Panel_Data['Page'] = eye_data.Panel
        Panel_Data['Page'] = eye_data.Page
        Panel_Data['IsBlink'] = Discard_Blink(pd.DataFrame(eye_data.IsBlink))
        Panel_Data.apply(lambda x:ResetPanel(x), axis = 1)
        eye_data['Panel'] = Discard_short_look(Panel_Data).reset_index(drop = True)
        eye_data['IsBlink'] = Panel_Data['IsBlink']
        eye_data.to_csv(path_or_buf = 'data/AddedHeatmap/' + str(ID) + '.csv', index = False)
        PastPage = -1
        PastPanel = -1
        ResetedPanel = pd.DataFrame()
        print(ID)

if __name__ == "__main__":
    main()