#!/usr/bin/env python3
# Check missing values and calculate the average values
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : Mar 25 2018
import sys
import csv
import numpy as np
import datetime
import pylab as plt

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

if __name__ == "__main__":
    # read data from csv file
    #path = '/Users/ylk/github/arm-pearson/netcdf_year_viz/E33_1993_2017.csv'
    path = '/Users/yupinglu/github/arm-pearson/netcdf_year_viz/E1_1993_2017.csv'
    begin = 1996
    end = 2009
    var_name = 'temp_mean'
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
    # print the hist of the residuals
    '''
    n, bins, patches = plt.hist(residuals)
    plt.show()
    '''
    # print the top 10 extreme values
    '''
    x = np.absolute(residuals)
    ix  = np.argsort(x)
    for i in range(len(ix)-1, len(ix)-11, -1):
        print(t[ix[i]].date(), residuals[ix[i]])
    '''
    # get 95% confidence interval, but the sample size is large. use 68–95–99.7 rule instead
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    #SE = sigma / np.sqrt(len(residuals))
    ci0 = mu - 3 * sigma
    ci1 = mu + 3 * sigma
    #print the outcomes
    #print('99.7% confidence inverval:', ci0, ci1, residuals.min(), residuals.max())
    #print the outliers
    for i in range(len(residuals)):
        if residuals[i] < ci0 or residuals[i] > ci1:
            print(t[i].date(), residuals[i])

    # plot the result
    fig,axs = plt.subplots(nrows=len(groups)+2,tight_layout=True)
    axs[0].plot(t,gpp,'-')
    axs[0].set_ylim(gpp.min(),gpp.max())    
    for g in range(len(groups)+1):
        axs[g].plot(t,decomp[g],'-')
    axs[-1].plot(t,gpp-decomp.sum(axis=0),'-')
    Y1 = [ci0] * len(t)
    Y2 = [ci1] * len(t)
    #axs[-1].plot(t,Y1,lw=1)
    #axs[-1].plot(t,Y2,lw=1)
    axs[-1].fill_between(t, Y1, Y2, alpha=0.5)
    axs[0].set_ylabel("Raw & Trend")
    axs[1].set_ylabel("Year")
    axs[2].set_ylabel("Month")
    axs[3].set_ylabel("Residual")
    plt.show()
    
    