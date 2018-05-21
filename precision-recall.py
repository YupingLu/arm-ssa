#!/usr/bin/env python3
# Calculate the combined precision and recall
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 21 2018

#load libs
import csv
import pandas as pd
import datetime
import os

# Get the whole dates 2
def getDates2(begin, end):
    x = []
    span = (end - begin).days + 1
    for i in range(span):
        x.append(begin + datetime.timedelta(i))
    return x

# Return a set of outliers
def readOutliers(dir, inst):
    outliers1 = []
    path = "/Users/ylk/github/arm-ssa/outliers/"+dir+"/E"+inst+'.txt'
    #path = "/Users/yupinglu/github/arm-ssa/outliers/"+dir+"/E"+inst+'.txt'
    if os.stat(path).st_size != 0:
        df = pd.read_csv(path, header=None)
        outliers = df[df.columns[0]]
        outliers1 = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in outliers]
    return set(outliers1)

# read DQR records
def readDB(inst):
    path = '/Users/ylk/github/arm-ssa/db.records/kmeans/E'+inst+'.db.csv'
    #path = '/Users/yupinglu/github/arm-ssa/db.records/kmeans/E'+inst+'.db.csv'
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

if __name__ == "__main__":
    insts = ['1','3','4','5','6','7','8','9','11','13','15','20','21','24','25','27','31','32','33',\
    '34','35','36','37','38']

    TP = 0 # True positive: outliers in DQR
    FP = 0 # False positive: outliers not in DQR
    FN = 0 # False negative: undetected values in DQR
    #TN = 0 # true negative: undetected values not in DQR
    
    for inst in insts:
        outliers = set() # store the dates of outliers
        dqr = set() # dqr records

        # read dqr records
        xs1, xs2 = readDB(inst)
        for idx in range(len(xs1)):
            dqr |= set(getDates2(xs1[idx], xs2[idx]))
        # read outliers
        outliers |= readOutliers('kmeans', inst)
        outliers |= readOutliers('ssa_atmos_pressure', inst)
        outliers |= readOutliers('ssa_rh_mean', inst)
        outliers |= readOutliers('ssa_temp_mean', inst)
        outliers |= readOutliers('ssa_vapor_pressure_mean', inst)
        outliers |= readOutliers('ssa_wspd_arith_mean', inst)
        '''
        if len(dqr) != 0:
            print(type(list(dqr)[0]))
        if len(outliers) != 0:
            print(type(list(outliers)[0]))
        '''
        tmp_tp = len(outliers & dqr)
        tmp_fp = len(outliers - dqr)
        tmp_fn = len(dqr - outliers)
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
    print("Precison: ", '{:.1%}'.format(P))
    print("Recall: ", '{:.1%}'.format(R))