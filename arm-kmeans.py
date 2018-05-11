#!/usr/bin/env python3
# Extract outliers using kmeans - multivariate
# Results are visualized in plotly
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 11 2018

#load libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go

# use plotly to visualize the data
def plotKmeansRes(inst):
    #path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'

    cols_to_use = [0,1,2,3,4,5]
    df = pd.read_csv(path, usecols=cols_to_use, na_values='None')
    # remove empty fields
    df.dropna(inplace=True)
    df.index = range(len(df))

    # standardization
    # axis is 0 (column) by default, independently standardize each feature
    data = df[['atmos_pressure','temp_mean','rh_mean','vapor_pressure_mean','wspd_arith_mean']]
    X = scale(data) 

    # run k-means
    model = KMeans(n_clusters=4)
    model.fit(X)

    # get the centroids
    '''
    data_std = np.asarray(np.std(data, axis=0))
    data_mean = np.asarray(np.mean(data, axis=0))
    print(model.cluster_centers_ * data_std + data_mean)
    '''

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

    layout = {'title':'E'+inst}

    plotly.offline.plot({
        "data": data,
        #"layout": go.Layout(title="test")
        "layout": layout

    }, filename ='E'+inst+'.html', show_link = False, auto_open = False)


if __name__ == "__main__":
    # read data from csv file
    insts = ['1','3','4','5','6','7','8','9','11','13','15','20','21','24','25','27','31','32','33',\
    '34','35','36','37','38']
    for inst in insts:
        plotKmeansRes(inst)