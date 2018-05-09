#!/usr/bin/env python3
# Extract outliers using kmeans - multivariate
# Results are visualized in plotly
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 09 2018

#load libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go

inst = '38'
#path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'

cols_to_use = [0,1,2,3,4,5]
df = pd.read_csv(path, usecols=cols_to_use, na_values='None')
# remove empty fields
df.dropna(inplace=True)
df.index = range(len(df))

# standardization
# axis is 0 (column) by default, independently standardize each feature
X = scale(df[['atmos_pressure','temp_mean','rh_mean','vapor_pressure_mean','wspd_arith_mean']]) 

# run k-means
model = KMeans(n_clusters=4)
model.fit(X)

# get the dates of 4 clusters
cls1 = df['date'][model.labels_ == 0]
cls2 = df['date'][model.labels_ == 1]
cls3 = df['date'][model.labels_ == 2]
cls4 = df['date'][model.labels_ == 3]

# transform X to cluster-distance space
X1 = model.transform(X)
dist = np.sum(X1, axis=1)

# the top 10 index of the longest distance
idx = dist.argsort()[-10:][::-1] 

# Visualize the results
trace1 = go.Scatter(
    x = cls1,
    y = dist[model.labels_ == 0],
    mode = 'markers',
    name = 'cluster 1'
)
trace2 = go.Scatter(
    x = cls2,
    y = dist[model.labels_ == 1],
    mode = 'markers',
    name = 'cluster 2'
)
trace3 = go.Scatter(
    x = cls3,
    y = dist[model.labels_ == 2],
    mode = 'markers',
    name = 'cluster 3'
)
trace4 = go.Scatter(
    x = cls4,
    y = dist[model.labels_ == 3],
    mode = 'markers',
    name = 'cluster 4'
)
trace5 = go.Scatter(
    x = df['date'][idx],
    y = dist[idx],
    mode = 'markers',
    marker=dict(
        size='10',
        color = 'Red',
        symbol = 'square'
    ),
    name = 'Outliers'
)
data = [trace1, trace2, trace3, trace4, trace5]

layout = {'title':'E38'}

plotly.offline.plot({
    "data": data,
    #"layout": go.Layout(title="test")
    "layout": layout

}, filename ='E38.html', show_link = False, auto_open = False)


