import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from uncertainties import ufloat
import scipy.stats as st
from scipy.optimize import fmin
from sympy import nsolve
from sympy import Symbol
from sklearn.utils import resample
from datetime import datetime
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
    plt.ylabel("Portion of People Who Have Tried At Least Once")
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



def invGammaSkew(p):
    a,b=p
    return (4(a-2)**(1./2)/(a-3))
def invGammaSkew(p):
    a,b=p
    6*(5*a-11)/((a-3)(a-4))

def specialInvGamma(mean,var):
    r=[]
    a=Symbol('a')
    b=Symbol('b')
    def invGammaMean(a, b):
        return b / (a - 1) - mean

    def invGammaVar(a, b):
        return (b ** 2) / ((a - 1) ** 2 * (a - 2)) - var
    a,b=nsolve([b/(a-1)-mean,(b**2)/((a-1)**2*(a-2))-var], [a,b], (4,4),tol=1.)
    print(a)
    print(b)
    return r

def invGammaFrom4Moments(df,names):
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    for name in names:
        df=uNormalizeColumn(df, name)
    l_kurt = df.kurtosis()[0::2]+3
    l_skew= df.skew()[0::2]**2
    l_variance=df.var()[0::2]
    l_mean = df.mean()[0::2]
    for mean, variance in zip(l_mean,l_variance):
        specialInvGamma(mean,variance)
    #a1,b1 = fsolve(lambda x: )

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
def kurtSkewBoot(df,names):
    for name in names:
        l_kurt = []
        l_skew2 = []
        df = uNormalizeColumn(df, name)
        l_kurt.append(df.kurtosis()[0::2]+3)
        l_skew2.append(df.skew()[0::2]**2)
        data=df[name+wcstr].values
        df_boot = pd.DataFrame()
        for i in range(0,40):
            df_boot[name+str(i)]=pd.Series(resample(data,replace=True,n_samples=len(data),random_state=int(datetime.now().timestamp())))
        l_kurt.append(df_boot.kurtosis() + 3)
        l_skew2.append(df_boot.skew()** 2)

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
    gammas= kurtSkew(df.copy(),[lsdNick,cocNick,emNick],["darkred","salmon","mistyrose","b"])
    #addGammas(df.index.values, mainPlot,gammas)
    #invGammaFrom4Moments(df,[lsdNick,cocNick,emNick])
    kurtSkewBoot(df,[lsdNick,cocNick,emNick])
if __name__ == '__main__':
    main()