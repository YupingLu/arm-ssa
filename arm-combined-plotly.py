#!/usr/bin/env python3
# Extract outliers using SSA method and kmeans
# Results are visualized in plotly
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date : June 9 2018

#load libs
import sys
import csv
import pandas as pd
import numpy as np
import datetime
import os
import plotly
import plotly.graph_objs as go

# Read a variable from a csv file, check missing values 
# Return a dict {date:variable}
def readCSVFile(path, name):
    res = {}
    #date,atmos_pressure,temp_mean,rh_mean,vapor_pressure_mean,wspd_arith_mean,tbrg_precip_total_corr
    cols_to_use = [0,1,2,3,4,5]
    df = pd.read_csv(path, usecols=cols_to_use, na_values='None')
    # remove empty fields
    df.dropna(inplace=True)
    df.index = range(len(df))
    for i in range(len(df)):
        key = datetime.datetime.strptime(df['date'][i], '%Y-%m-%d %H:%M:%S')
        res[key] = df[name][i]
    return res

# Return a set of outliers
def readOutliers(dir, inst):
    outliers1 = []
    #path = "/Users/ylk/github/arm-ssa/outliers/"+dir+"/E"+inst+'.txt'
    path = "/Users/yupinglu/github/arm-ssa/outliers/"+dir+"/E"+inst+'.txt'
    if os.stat(path).st_size != 0:
        df = pd.read_csv(path, header=None)
        outliers = df[df.columns[0]]
        outliers1 = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in outliers]
    return set(outliers1)

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
def plotRes(inst, var_name, idx):
    #path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    #path1 = '/Users/ylk/github/arm-ssa/db.records/'+var_name+'/E'+inst+'.db.csv'
    path1 = '/Users/yupinglu/github/arm-ssa/db.records/'+var_name+'/E'+inst+'.db.csv'
    var_dict = readCSVFile(path, var_name)
    
    ko = set() # store the kmeans outliers
    so = set() # store the ssa outliers
    ko |= readOutliers('kmeans', inst)
    so |= readOutliers('ssa_atmos_pressure', inst)
    so |= readOutliers('ssa_rh_mean', inst)
    so |= readOutliers('ssa_temp_mean', inst)
    so |= readOutliers('ssa_vapor_pressure_mean', inst)
    so |= readOutliers('ssa_wspd_arith_mean', inst)

    # check missing values and replace it with 0
    lm = list(ko | so)
    for key in lm:
        if key not in var_dict:
            var_dict[key] = 0

    res = []
    t = []
    for key in sorted(var_dict):
        t.append(key)
        res.append(var_dict[key])
    gpp = np.asarray(res, dtype=np.float32)

    x_inter = list(ko & so)
    y_inter = []
    for i in range(len(x_inter)):
        y_inter.append(var_dict[x_inter[i]])

    x_koonly = list(ko - so)
    y_koonly = []
    for i in range(len(x_koonly)):
        y_koonly.append(var_dict[x_koonly[i]])

    x_soonly = list(so - ko)
    y_soonly = []
    for i in range(len(x_soonly)):
        y_soonly.append(var_dict[x_soonly[i]])
    
    # plot the result
    trace1 = go.Scatter(
        x = t,
        y = gpp,
        mode = 'lines',
        name = var_name
        )
    trace2 = go.Scatter(
        x = x_inter,
        y = y_inter,
        mode = 'markers',
        marker=dict(
            size='10',
            color = 'Red',
            symbol = 'square'
        ),
        name = 'kmeans & ssa'
    )
    trace3 = go.Scatter(
        x = x_koonly,
        y = y_koonly,
        mode = 'markers',
        marker=dict(
            size='10',
            color = 'Orange',
            symbol = 'diamond'
        ),
        name = 'kmeans only'
    )
    trace4 = go.Scatter(
        x = x_soonly,
        y = y_soonly,
        mode = 'markers',
        marker=dict(
            size='10',
            color = 'Black',
            symbol = 'star'
        ),
        name = 'ssa only'
    )
    data = [trace1, trace2, trace3, trace4]
    # plot DQR records with shade regions
    xs1, xs2 = readDB(path1)
    layout = {'shapes':[], 'title':'E'+inst+'-'+var_name}
    for i in range(len(xs1)):
        shape = {}
        shape['type'] = 'rect'
        shape['xref'] = 'x'
        shape['yref'] = 'paper'
        shape['x0'] = xs1[i]
        shape['y0'] = 0
        shape['x1'] = xs2[i]
        shape['y1'] = 1
        shape['fillcolor'] = '#d3d3d3'
        shape['opacity'] = 0.2
        shape['line'] = {}
        shape['line']['width'] = 0
        layout['shapes'].append(shape)
    plotly.offline.plot({
        "data": data,
        "layout": layout
    }, filename = 'E'+inst+'.'+str(idx)+'.html', show_link = False, auto_open = False)    

if __name__ == "__main__":
    inst = ['1','3','4','5','6','7','8','9','11','13','15','20','21','24','25','27','31','32','33',\
    '34','35','36','37','38']
    var_names = ['temp_mean', 'vapor_pressure_mean', 'atmos_pressure', 'rh_mean', 'wspd_arith_mean']
        
    for i in range(len(inst)):
        for j in range(len(var_names)):
            plotRes(inst[i], var_names[j], j)
