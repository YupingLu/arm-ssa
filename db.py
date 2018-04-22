#!/usr/bin/env python3
# Query DQR databse
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date  : April 22 2018

#load libs
import psycopg2
import datetime
import numpy as np

# Get the whole dates
def getDates(begin, end):
    x = []
    span = (end - begin).days + 1
    for i in range(span):
        x.append(begin + datetime.timedelta(i))
    return x

# Execute sql and return the dates
def query(sql):
    begin = []
    end = []
    try:
        conn = psycopg2.connect("dbname='arm_all' user='yok' host='armdev-pgdb.ornl.gov' password='yok111'")
    except:
        print("Unable to connect to the database")
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        begin.append(row[0].date())
        if row[1] == None:
            end.append(row[0].date())
        else:
            end.append(row[1].date())
    conn.close()
    return begin, end

# save sql query results
def saveRes(inst):
     # sql command
    sql = """
        SELECT
            vm.start_date,
            vm.end_date
        FROM
            pifcardqr2.varname_metric vm
        INNER JOIN pifcardqr2.dqr dqr on
            dqr.dqrid = vm.id
        WHERE
            vm.datastream = 'sgpmetE"""+inst+""".b1'
            AND vm.var_name = 'temp_mean'
            AND description NOT IN(
                'Test',
                'test',
                'REQUIRED'
            );"""
    begin, end = query(sql)
    # save results to csv file
    res = []
    res.append(begin)
    res.append(end)
    res_name = "E"+inst+".db.csv"
    np.savetxt(res_name, np.transpose(res), delimiter=",", comments="", fmt='%s', \
            header="start_date, end_date")

if __name__ == "__main__":
    insts = [1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 20, 21, 24, 25, 27, 31, 32, 33, 34, 35, 36, 37, 38]
    for inst in insts:
        saveRes(str(inst))
   