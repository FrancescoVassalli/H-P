import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from uncertainties import ufloat
import scipy.stats as st

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
    plt.figure(num=1,figsize=( 16,12),facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    for name, color, label in zip(names,colors,plotlabels):
        df = uNormalizeColumn(df,name)
        plt.scatter(df.index.values,df[name+wcstr].values,color=color, s=100, marker='^',label=label)
        plt.errorbar(df.index.values,df[name+wcstr].values,yerr=df[name+sestr].values, color=color,linestyle='None')
    ax.legend()
    plt.xlabel("Age First Tried")
    plt.ylabel("Fraction of People Who Have Tried")
    plt.ylim(0,.15)
    plt.show()

def specialGamma(skew2,var):
    r=[]
    for sw, v in zip(skew2,var):
        alpha=( sw /4)**-1
        beta = (alpha/v)**(1./2)
        print('a:'+str(alpha)+' b:'+str(beta))
        r.append(st.gamma(alpha, scale=beta))
    return r
def specialInvGamma(skew2,var):
    r=[]
    for sw, v in zip(skew2,var):
        alpha=( sw /4)**-1
        beta = (alpha/v)**(1./2)
        print('a:'+str(alpha)+' b:'+str(beta))
        r.append(st.gamma(alpha, scale=beta))
    return r

def invGammaFrom4Moments(df,names):
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    for name in names:
        df=uNormalizeColumn(df, name)
    print("Kurtosis:")
    kurt = df.kurtosis()[0::2]+3
    print(kurt)
    print("Skew:")
    skew= df.skew()[0::2]**2
    print(skew)
    print("Variance:")
    variance=df.var()[0::2]
    print((variance))
    print("Mean:")
    mean = df.mean()[0::2]
    print((mean))

def kurtSkew(df,names,colors):
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    for name in names:
        df=uNormalizeColumn(df, name)
    print("Kurtosis:")
    kurt = df.kurtosis()[0::2]+3
    print(kurt)
    print("Skew:")
    skew= df.skew()[0::2]**2
    print(skew)
    plt.scatter(skew,kurt,color=colors)
    plt.show()
    print(np.poly1d(np.polyfit(skew,kurt,1)))
    return specialGamma(skew,df.var()[0::2])

def addGammas(xVals,mainPlot,gammas):
    fig, ax = plt.subplots()
    for gamma in gammas:
        ax.plot(xVals,gamma.pdf(xVals))
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
    str_her = "AGE WHEN FIRST USED HEROIN"
    her_path = r"C:\Users\Francesco Vassalli\Downloads\HERage.csv"
    herNick = 'Her: '
    df=df.join([makeDrugFrame(str_coc, cocNick, coc_path),makeDrugFrame(str_em,emNick,em_path),makeDrugFrame(str_her,herNick,her_path)])
    #mainPlot=makeMainPlot(df.copy(),[lsdNick,cocNick,emNick,herNick],["darkred","salmon","mistyrose",'b'],["LSD","Cocaine","Ecstasy/Molly","Heroin"])
    #gammas= kurtSkew(df.copy(),[lsdNick,cocNick,emNick],["darkred","salmon","mistyrose","b"])
    #addGammas(df.index.values, mainPlot,gammas)
    invGammaFrom4Moments(df,[lsdNick,cocNick,emNick])
if __name__ == '__main__':
    main()