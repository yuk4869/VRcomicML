# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:01:03 2021

@author: sakam
"""

import pandas as pd
import numpy as np

SkipID = [4, 6, 10, 11, 12, 13, 14, 15, 16]
TileList = pd.DataFrame()
TileListTemp = pd.DataFrame()

def JudgeTile(_X, _Y):
    if (abs(_X) > 577.0) | (abs(_Y) > 750.0):
        return -1
    
    if _X > 0:
        if _Y >= 0:
            return 1
        else:
            return 3
    else:
        if _Y >= 0:
            return 2
        else:
            return 4
        
def MakeTileLIst(row_data):
    global TileListTemp
    TileListTemp = pd.concat([TileListTemp, pd.DataFrame([JudgeTile(row_data.GazePointX, row_data.GazePointY)])])

def Discard_While_Blink():
    global TileList
    _pastTile = -1
    filterdTile = pd.DataFrame()

    for j, page in TileList.groupby([(TileList.Page != TileList.Page.shift()).cumsum()]):
        _pastTile = -1
        for k, blink in page.groupby([(page.IsBlink != page.IsBlink.shift()).cumsum()]):
            if blink.IsBlink.sum() != 0:
                blink.Tile = _pastTile
            filterdTile = pd.concat([filterdTile, blink.Tile])
            _pastTile = blink.Tile.iloc[-1]
    TileList.Tile = filterdTile.reset_index(drop = True)

def Discard_Short_Look():
    global TileList
    _pastTile = -1
    filterdTile = pd.DataFrame()
    
    for j, page in TileList.groupby([(TileList.Page != TileList.Page.shift()).cumsum()]):
        _pastTile = -1
        for k, tile in page.groupby([(page.Tile != page.Tile.shift()).cumsum()]):
            if (len(tile) < MIN_LOOK) | (tile.Tile.sum() < 0):
                tile.Tile = _pastTile
            filterdTile = pd.concat([filterdTile, tile.Tile])
            _pastTile = tile.Tile.iloc[-1]
                
    TileList.Tile = filterdTile.reset_index(drop = True)

MIN_LOOK = 6

def main():
    global TileList
    global TileListTemp
    for ID in range(2,26, 1):
        if ID in SkipID: continue
        eye_data = pd.read_csv('data/AddedHeatmap/' + str(ID) + '.csv', encoding="ms932" ,sep=",")
        TileList['Page'] = eye_data.Page
        TileList['IsBlink'] = eye_data.IsBlink
        eye_data.apply(lambda x:MakeTileLIst(x), axis = 1)
        TileList['Tile'] = TileListTemp.reset_index(drop = True)[0]
        Discard_While_Blink()
        Discard_Short_Look()
        eye_data['Tile'] = TileList.Tile.reset_index(drop = True)
        eye_data.to_csv(path_or_buf = 'data/AddedHeatmap/' + str(ID) + '.csv', index = False)
        print(ID)
        TileList = pd.DataFrame()
        TileListTemp = pd.DataFrame()

if __name__ == "__main__":
    main()