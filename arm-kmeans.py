#!/usr/bin/env python3
# Extract outliers using kmeans - multivariate
# Results are visualized in plotly
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 11 2018
# Add precison and recall calculation
# Date : may 21, 2018

#load libs
import csv
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go

# read DQR records
def readDB(path):
    xs1 = []
    xs2 = []
    # start_date, end_date
    # read all data
    with open( path, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            begin = datetime.datetime.strptime(line['start_date'], '%Y-%m-%d')
            end = datetime.datetime.strptime(line['end_date'], '%Y-%m-%d')
            xs1.append(begin)
            xs2.append(end)
    return xs1, xs2

# use plotly to visualize the data
def plotKmeansRes(inst):
    path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    #path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    path1 = '/Users/ylk/github/arm-ssa/db.records/kmeans/E'+inst+'.db.csv'
    #path1 = '/Users/yupinglu/github/arm-ssa/db.records/kmeans/E'+inst+'.db.csv'

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
    # get 95% confidence interval, but the sample size is large. use 68–95–99.7 rule instead
    mu = np.mean(dist)
    sigma = np.std(dist)
    ci1 = mu + 3 * sigma
    x_t = []
    y_outliers = []
    for i in range(len(dist)):
        if dist[i] > ci1:
            x_t.append(df['date'][i])
            y_outliers.append(dist[i])
    # the top 10 index of the longest distance
    #idx = dist.argsort()[-10:][::-1] 

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
        x = x_t,
        y = y_outliers,
        mode = 'markers',
        marker=dict(
            size='10',
            color = 'Red',
            symbol = 'square'
        ),
        name = 'Outliers'
    )
    data = [trace1, trace2, trace3, trace4, trace5]
    xs1, xs2 = readDB(path1)
    layout = {'title':'E'+inst}
    '''
    plotly.offline.plot({
        "data": data,
        #"layout": go.Layout(title="test")
        "layout": layout

    }, filename ='E'+inst+'.html', show_link = False, auto_open = False)
    '''
    return x_t, xs1, xs2

# Get the whole dates 2
def getDates2(begin, end):
    x = []
    span = (end - begin).days + 1
    for i in range(span):
        x.append(begin + datetime.timedelta(i))
    return x

if __name__ == "__main__":
    # read data from csv file
    insts = ['1','3','4','5','6','7','8','9','11','13','15','20','21','24','25','27','31','32','33',\
    '34','35','36','37','38']

    TP = 0 # True positive: outliers in DQR
    FP = 0 # False positive: outliers not in DQR
    FN = 0 # False negative: undetected values in DQR
    #TN = 0 # true negative: undetected values not in DQR

    for inst in insts:
        x_t, xs1, xs2 = plotKmeansRes(inst)

        dqr = set()  # dqr records
        ks = set(x_t)  # outliers using kmeans
        np.savetxt('E'+str(inst)+'.txt', list(ks), delimiter=",", comments="", fmt='%s')

        for idx in range(len(xs1)):
            dqr |= set(getDates2(xs1[idx], xs2[idx]))

        tmp_tp = len(dqr & ks)
        tmp_fp = len(ks - dqr)
        tmp_fn = len(dqr - ks)
        TP += tmp_tp
        FP += tmp_fp
        FN += tmp_fn
        if tmp_tp + tmp_fp == 0:
            print("E"+str(inst)+" precison is empty.")
        else:
            p = tmp_tp / (tmp_tp + tmp_fp)
            print("E"+str(inst)+" precison: ", '{:.1%}'.format(p))
        if tmp_tp + tmp_fn == 0:
            print("E"+str(inst)+" recall is empty.")
        else:
            r = tmp_tp / (tmp_tp + tmp_fn)
            print("E"+str(inst)+" recall: ", '{:.1%}'.format(r))

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("kmeans precison: ", '{:.1%}'.format(P))
    print("kmeans recall: ", '{:.1%}'.format(R))
