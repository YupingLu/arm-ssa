#!/usr/bin/env python3
# Extract outliers using SSA method
# Results are visualized in plotly
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date : April 30 2018
# Add precison and recall calculation
# Date : may 19, 2018

#load libs
import sys
import csv
import numpy as np
import datetime
import plotly
import plotly.graph_objs as go

def SSA(Y,L,period_groups):
    T = Y.size
    assert L <= T/2
    K = T - L + 1
    # Form the trajectory matrix and find the eigen decomp
    X = np.zeros((L,K))
    for i in range(K): X[:,i] = Y[i:(i+L)]
    lamda,P = np.linalg.eig(np.dot(X,X.T))
    # Find the dominant frequency of each eigenvector
    f   = np.zeros(lamda.size)
    fs  = np.fft.fftfreq(f.size,1.)
    ix  = np.argsort(fs)
    fs  = fs[ix]
    eps = 0.99*(fs[1]-fs[0])
    for i in range(f.size):
        ps = np.abs(np.fft.fft(P[:,i]))**2
        ps = ps[ix]
        f[i] = fs[ps.argmax()]
    f = np.abs(f)
    # convert periodicity into frequency
    fgroups = 1/np.asarray(period_groups,dtype=float)
    fgroups = np.hstack([0,fgroups])
    # Build an approximation of X by taking a subset of the
    # decomposition. This approximation is formed by taking
    # eigenvectors whose dominant frequency is close to the targetted
    # values.
    Xt = np.zeros((fgroups.size,)+X.shape)
    for i in range(f.size):
        g = np.where(np.abs(fgroups-f[i]) < eps)[0]
        if g.size == 0: continue
        Xt[g[0]] += np.dot(np.outer(P[:,i],P[:,i]),X)
    # Now we reconstruct the signal by taking a mean of all the
    # approximations.
    Yt = np.zeros((fgroups.size,Y.size))
    c  = np.zeros((fgroups.size,Y.size))
    for g in range(fgroups.size):
        for i in range(K): 
            Yt[g,i:(i+L)] += Xt[g,:,i]
            c [g,i:(i+L)] += 1
    Yt /= c
    return Yt

# Get the whole dates
def getDates(byear, eyear):
    x = []
    start = datetime.date(byear-1, 12, 31)
    end = datetime.date(eyear, 12, 31)
    span = (end - start).days
    begin = datetime.datetime(byear, 1, 1, 0, 0)
    for i in range(span):
        x.append(begin + datetime.timedelta(i))
    return x

# Read a variable from a csv file, check missing values 
# And replace missing values with average value
# Return a dict {date:variable}
def readCSVFile(path, name, begin, end):
    res = {}
    begin_date = datetime.datetime(begin, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(end, 1, 1, 0, 0, 0)
    # read all data
    with open( path, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            key = datetime.datetime.strptime(line['date'], '%Y-%m-%d %H:%M:%S')
            if line[name] != 'None' and key >= begin_date and key < end_date:
                res[key] = float(line[name])
    # compute average values
    cnt = [0] * 366
    average = [0] * 366
    for i in range(begin, end):
        dates = getDates(i, i)
        count = -1
        for date in dates:
            count += 1
            if date in res:
                cnt[count] += 1
                average[count] += res[date]
    for i in range(len(cnt)):
        if cnt[i] != 0:
            average[i] /= cnt[i]
        else:
            average[i] = -40            
    # replace missing values with average ones
    for i in range(begin, end):
        dates = getDates(i, i)
        count = -1
        for date in dates:
            count += 1
            if date not in res:
                res[date] = average[count]
    return res

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
def plotRes(inst, begin, end, var_name):
    #path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E'+inst+'_1993_2017.csv'
    #path1 = '/Users/ylk/github/arm-ssa/db.records/'+var_name+'/E'+inst+'.db.csv'
    path1 = '/Users/yupinglu/github/arm-ssa/db.records/'+var_name+'/E'+inst+'.db.csv'
    
    var_dict = readCSVFile(path, var_name, begin, end)
    # compute SSA and extract residuals
    res = []
    t = []
    for key in sorted(var_dict):
        t.append(key)
        res.append(var_dict[key])
    gpp = np.asarray(res, dtype=np.float32)
    groups = [365, 30]
    decomp = SSA(gpp,400,groups)
    # output the extream values
    residuals = gpp-decomp.sum(axis=0)
    # get 95% confidence interval, but the sample size is large. use 68–95–99.7 rule instead
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    #SE = sigma / np.sqrt(len(residuals))
    ci0 = mu - 3 * sigma
    ci1 = mu + 3 * sigma
    #print the outcomes
    #print('99.7% confidence inverval:', ci0, ci1, residuals.min(), residuals.max())
    #print the outliers
    x_t = []
    y_outliers = []
    for i in range(len(residuals)):
        if residuals[i] < ci0 or residuals[i] > ci1:
            #print(t[i].date(), residuals[i])
            x_t.append(t[i])
            y_outliers.append(gpp[i])
    # plot the result
    trace1 = go.Scatter(
        x = t,
        y = gpp,
        mode = 'lines',
        name = var_name
        )
    trace2 = go.Scatter(
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
    data = [trace1, trace2]
    #data = [trace1, trace2, trace3, trace4]
    # plot DQR records with shade regions
    xs1, xs2 = readDB(path1)
    layout = {'shapes':[], 'title':'E'+inst+'-'+str(begin)+'-'+str(end-1)}
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
    '''
    plotly.offline.plot({
        "data": data,
        "layout": layout

    }, filename = 'E'+inst+'-'+str(begin)+'-'+str(end-1)+'.html', show_link = False, auto_open = False)
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
    inst = ['1','3','4','5','6','7','8','9','11','13','15','20','21','24','25','27','31','32','33',\
    '34','35','36','37','38']
    begin = [1996,1997,1996,1997,1997,1996,1994,1994,1996,1994,1994,1994,2000,1996,1997,2004,2012,\
    2012,2012,2012,2012,2012,2012,2012]
    end = [2009,2009,2011,2009,2011,2012,2009,2018,2018,2018,2018,2011,2018,2009,2002,2010,2018,\
    2018,2018,2018,2018,2018,2018,2018]
    # switch variable here. (temp_mean, vapor_pressure_mean, atmos_pressure, rh_mean, wspd_arith_mean)
    var_name = 'wspd_arith_mean'
    
    TP = 0 # True positive: outliers in DQR
    FP = 0 # False positive: outliers not in DQR
    FN = 0 # False negative: undetected values in DQR
    #TN = 0 # true negative: undetected values not in DQR
    outliers = set() # store the dates of outliers

    for i in range(len(inst)):
        x_t, xs1, xs2 = plotRes(inst[i], begin[i], end[i], var_name)

        dqr = set()  # dqr records
        ssa = set(x_t)  # outliers using ssa
        outliers |= ssa # set union

        for idx in range(len(xs1)):
            dqr |= set(getDates2(xs1[idx], xs2[idx]))

        tmp_tp = len(dqr & ssa)
        tmp_fp = len(dqr - ssa)
        tmp_fn = len(ssa - dqr)
        TP += tmp_tp
        FP += tmp_fp
        FN += tmp_fn
        if tmp_tp + tmp_fp == 0:
            print("E"+str(inst[i])+" precison is empty.")
        else:
            p = tmp_tp / (tmp_tp + tmp_fp)
            print("E"+str(inst[i])+" precison: ", '{:.1%}'.format(p))
        if tmp_tp + tmp_fn == 0:
            print("E"+str(inst[i])+" recall is empty.")
        else:
            r = tmp_tp / (tmp_tp + tmp_fn)
            print("E"+str(inst[i])+" recall: ", '{:.1%}'.format(r))

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("SSA precison: ", '{:.1%}'.format(P))
    print("SSA recall: ", '{:.1%}'.format(R))
    np.savetxt('ssa_outliers.txt', list(outliers), delimiter=",", comments="", fmt='%s')