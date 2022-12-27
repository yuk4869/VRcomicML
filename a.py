# SkipID = [3, 4, 6, 10, 11, 12, 13, 14, 15, 16, 19, 21, 23]
# for ID in range(2, 6, 1): #26 to 6
#         if ID not in SkipID:continue
#         print(ID)
#         if ID in SkipID:print("adfafads")

import pandas as pd
import numpy as np

import time

df = pd.DataFrame()
df["x"] = np.arange(1e8)

start = time.time()
df["x"] += 1
df["y"] = df["x"] * 2
df["z"] = df["x"] / df["y"]
end = time.time()
print(end - start)

df = pd.DataFrame()
df["x"] = np.arange(1e8)
start = time.time()
df.eval(
        """
        x = x + 1
        y = x * 2
        z = x / y
        """, inplace=True
)
end = time.time()

print(end - start)