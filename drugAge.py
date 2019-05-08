import pandas as pd
import numpy as np
import math
import seaborn as  sns
import matplotlib.pyplot as plt
#from sklearn import preprocesssing
from uncertainties import ufloat

wcstr = "Weighted Count"
sestr = "Count SE"

def makeDrugFrame(drugName, nickname, filepath):
    agerange = range(10, 66)
    tried_df = pd.read_csv(filepath)
    tried_df = tried_df[[drugName, wcstr, sestr]]
    tried_df = tried_df[tried_df[drugName].apply(lambda x: x.isnumeric() and int(x) in agerange)]
    tried_df[drugName] = tried_df[drugName].astype(int)
    tried_df.set_index(drugName, inplace=True)
    tried_df.sort_index(inplace=True)
    tried_df.rename(columns=lambda x: nickname+ x, inplace=True)
    return tried_df

def makeDistPlot(df,nickname,color):
    plot = df[nickname+wcstr].plot(kind='bar',yerr=df[nickname+' '+sestr])
    return plot
def uNormalizeColumn(df,name):
    sum = ufloat(0., 0.)
    for i in df.index:
        count = ufloat(df.loc[i, name + wcstr], df.loc[i, name + sestr])
        if math.isnan(df.loc[i, name + wcstr]):
            count = ufloat(0., 0.)
        sum = sum + count
    for i in df.index:
        count = ufloat(df.loc[i, name + wcstr], df.loc[i, name + sestr])
        if math.isnan(df.loc[i, name + wcstr]):
            count = ufloat(0., 0.)
        else:
            count = count / sum
        df.loc[i, name + wcstr] = count.n
        df.loc[i, name + sestr] = count.s
    return df

def makeMainPlot(df,names,colors,plotlabels):
    plots=[]
    bottom=(0)*41
    fig, ax = plt.subplots()
    for name, color, label in zip(names,colors,plotlabels):
        df = uNormalizeColumn(df,name)
        plots.append(df[name+wcstr].plot(yerr=df[name+sestr],color=color,figsize=(15,7),label=label,legend=True,ax=ax))
        #bottom=bottom+ df[name+wcstr].values
    plt.show()

def main():
    str_lsd = "AGE WHEN FIRST USED LSD"
    lsd_path = r"C:\Users\Francesco Vassalli\Downloads\LSDage.csv"
    lsdNick = 'LSD: '
    df=makeDrugFrame(str_lsd, lsdNick, lsd_path)
    df.index.names = ['Age']
    df=df.reindex(pd.Index(np.arange(10,51,1),name='Age'))
    str_coc = "AGE WHEN FIRST USED COCAINE"
    coc_path = r"C:\Users\Francesco Vassalli\Downloads\COCage.csv"
    cocNick = 'COC: '
    str_em = "AGE WHEN FIRST USED ECSTASY OR MOLLY"
    em_path = r"C:\Users\Francesco Vassalli\Downloads\ECSTMOAGE.csv"
    emNick = 'E/M: '
    df=df.join([makeDrugFrame(str_coc, cocNick, coc_path),makeDrugFrame(str_em,emNick,em_path)])
    makeMainPlot(df.copy(),[lsdNick,cocNick,emNick],["darkred","salmon","mistyrose"],["LSD","Cocaine","Ecstasy/Molly"])
    print(df.head())
    #makeMainPlot(df,[lsdNick,cocNick,emNick])
    #lsd_dist_plot = makeDistPlot(df,lsdNick,'darkred')
    #coc_dist_plot = makeDistPlot(df,cocNick,'salmon')
    #em_dist_plot = makeDistPlot(df,emNick,'mistyrose')
    #plt.show()

if __name__ == '__main__':
    main()