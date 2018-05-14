#!/usr/bin/env python3
# Extract outliers using graph method - multivariate
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 14 2018

#load libs
import numpy as np
import pandas as pd
import subprocess
import shlex
from sklearn.preprocessing import scale

# run paraclique code and return outliers
def run_pc(command, size):
    res = []
    tps = [False] * size
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf8')
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            tmp = output.strip().split('\t')
            for t in tmp:
                tps[int(t)] = True
    for i in range(len(tps)):
        if tps[i] == False:
            res.append(i)
    return res

inst = '33'
path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
#path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'

cols_to_use = [0,1,2,3,4,5]
df = pd.read_csv(path, usecols=cols_to_use, na_values='None')

# remove empty fields
df.dropna(inplace=True)
df.index = range(len(df))

# standardization
# axis is 0 (column) by default, independently standardize each feature
data = df[['atmos_pressure','temp_mean','rh_mean','vapor_pressure_mean','wspd_arith_mean']]
X = scale(data) 

# calculate pearson correlation
pc = np.corrcoef(X)
apc = np.absolute(pc)

# extract absolute pc below 0.5
edges = 0
vertices = len(apc)
left = []
right = []
res = [False] * len(apc)
for i in range(1,len(apc)):
    for j in range(i-1):
        if apc[i][j] >= 0.5:
            edges += 1
            res[i] = True
            res[j] = True
            left.append(i)
            right.append(j)
for r in res:
    if r == False:
        vertices -= 1

# write the edgelist graph to a file
left.insert(0, vertices)
right.insert(0, edges)
cdf = []
cdf.append(left)
cdf.append(right)
np.savetxt('tmp.txt', np.transpose(cdf), delimiter="\t", comments="", fmt='%u')

# call paraclique code and print outliers
command = './paracl_cp tmp.txt 1 5 5 999999'
res = run_pc(command, len(apc))
for r in res:
    print(df['date'][r])
