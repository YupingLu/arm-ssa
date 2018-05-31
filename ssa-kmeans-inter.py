#!/usr/bin/env python3
# Calculate the intersection of outliers from ssa and kmeans
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : May 31 2018

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

    ko = set() # store the kmeans outliers
    so = set() # store the ssa outliers
    
    for inst in insts:
        # read outliers
        ko |= readOutliers('kmeans', inst)
        so |= readOutliers('ssa_atmos_pressure', inst)
        so |= readOutliers('ssa_rh_mean', inst)
        so |= readOutliers('ssa_temp_mean', inst)
        so |= readOutliers('ssa_vapor_pressure_mean', inst)
        so |= readOutliers('ssa_wspd_arith_mean', inst)
    
    print("Kmeans outlier size: ", len(ko))
    print("SSA outlier size: ", len(so))
    print("Intersection: ", len(ko & so))
    print("Symmetric difference: ", len(ko ^ so))