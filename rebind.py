import pandas as pd
from os import walk

def rebinds(dir):
    for root, dirs, files, in walk(dir, topdown=False):
        print(files)
    num = len(files)
    df = pd.DataFrame()
    for i in range(num):
        newdata = pd.read_excel(dir + r'/%s' %files[i]) # 读取excel
        df = df.append(newdata)
    writer = pd.ExcelWriter(dir + r'/output.xlsx')
    df.to_excel(writer, 'AllData')
    writer.save()