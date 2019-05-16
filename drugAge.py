import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from uncertainties import ufloat
import scipy.stats as st
from scipy.optimize import fmin
from scipy.optimize import curve_fit
from sympy import nsolve
from sympy import solve
from sympy import Symbol
from sklearn.utils import resample
from datetime import datetime
wcstr = "Weighted Count"
sestr = "Count SE"
gammabounds = ([0.,0.],['inf','inf'])

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

def invGammaCullenFreySlope(a):
    return 6.*(5*a-11)*(a-3)/(16*(a-2)*(a-4))

def invGammaMean(a, b):
    return b / (a - 1)
def invGammaVariance(a, b):
    return (b**2)/(((a-1)**2)*(a-2))



def kurtSkewBoot(df,names,colors):
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    l_slopes =[]
    for name, color in zip(names,colors):
        l_kurt = []
        l_skew2 = []
        df = uNormalizeColumn(df, name)
        #l_kurt.append(df.kurtosis()[0::2]+3)
        #l_skew2.append(df.skew()[0::2]**2)
        data=df[name+wcstr].values
        df_boot = pd.DataFrame()
        for i in range(0,40):
            df_boot[name+str(i)]=pd.Series(resample(data,replace=True,n_samples=len(data),random_state=i))
        l_kurt.append(df_boot.kurtosis() + 3)
        l_skew2.append(df_boot.skew()** 2)
        eq = np.poly1d(np.polyfit(np.asarray(l_skew2)[0], np.asarray(l_kurt)[0], 1))
        l_slopes.append(eq[0])
        print(eq)
        plt.scatter(np.asarray(l_skew2), np.asarray(l_kurt), color=color)
    plt.ylim(13,0)
    plt.show()
    return l_slopes

def getMeans(ndf,names):
    r = {}
    for name in names:
        r[name] = sum(np.array(ndf[name+wcstr].values)*np.array(ndf.index.values))
    return r
def getVariances(ndf,names,means):
    r = {}
    for name in names:
        r[name] = sum(np.array(ndf[name+wcstr].values)*(np.array(ndf.index.values)**2))-(means[name]**2)
    return r
def getSkewes(ndf,names,means,vars):
    r={}
    for name in names:
        r[name] = sum(np.array(ndf[name+wcstr].values)*(((np.array(ndf.index.values)-means[name])/vars[name])**3))
    return r

def getKurts(ndf,names,means):
    r={}
    for name in names:
        r[name] = sum((np.array(ndf[name+wcstr].values)*np.array(ndf.index.values)-means[name])**4)/(sum((np.array(ndf[name+wcstr].values)*np.array(ndf.index.values)-means[name])**2)**2)
    return r

def getFourMoments(ndf,names):
    moments = [getMeans(ndf, names)]
    moments.append(getVariances(ndf, names, moments[0]))
    moments.append(getSkewes(ndf, names, moments[0], moments[1]))
    moments.append(getKurts(ndf, names, moments[0]))
    return moments

def cullenFrey(ndf,names,colors):
    moments = getFourMoments(ndf,names)
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    for name,color in zip(names,colors):
        plt.scatter(moments[2][name]**2,moments[3][name],color=color)
    plt.ylim(.05,0)
    plt.xlim(0,.001)
    plt.show()

def specialInvGamma(mean,var):
    a = Symbol('a')
    b = Symbol('b')
    solutions=solve([invGammaMean(a,b)-mean,invGammaVariance(a,b)-var],[a,b],dict=True)[0]
    return float(solutions[a]),float(solutions[b])

def invGamma(df,names,colors):
    for name in names:
        df= uNormalizeColumn(df,name)
    l_mean = getMeans(df,names)
    l_var = getVariances(df,names,l_mean)
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    xVals= df.index.values
    for name,color in zip(names,colors):
        a,b = specialInvGamma(l_mean[name],l_var[name])
        ax.plot(xVals,st.invgamma.pdf(xVals,a,scale=b),color=color)
    plt.show()
def myDistFit(df,names,dist,bounds):
    r=[]
    for name in names:
        df[name+wcstr]=df[name+wcstr].fillna(0)
        df[name+sestr]=df[name+sestr].fillna(3000.)
        r.append(curve_fit(dist,df.index.values,df[name+wcstr].values,sigma=df[name+sestr],absolute_sigma=True,bounds=bounds))
    return r

def myGamma(x,a,b):
    return st.gamma.pdf(x,a,scale=b)
def myInvGamma(x,a,b):
    return st.invgamma.pdf(x,a,scale=b)
def myLogNormal(x,mu,sig):
    return st.lognorm.pdf(x,sig,loc=mu)

def addDistPlot(df,names,colors,dist, bounds):
    l_params=myDistFit(df,names,dist,bounds)
    plt.figure(num=1, facecolor='darkgray')
    fig, ax = plt.subplots(num=1)
    ax.set_facecolor('darkgray')
    xVals = df.index.values
    for param, color in zip(l_params,colors):
        print(param[0][0])
        ax.plot(xVals,dist(xVals,param[0][0],param[0][1]),color=color)
    plt.show()

def main():
    str_lsd = "AGE WHEN FIRST USED LSD"
    lsd_path = r"C:\Users\Francesco Vassalli\Downloads\LSDage.csv"
    lsdNick = 'LSD: '
    df=makeDrugFrame(str_lsd, lsdNick, lsd_path)
    df.index.names = ['Age']
    df.index = df.index.astype(int)
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
    names = [lsdNick,cocNick,emNick,herNick]
    colors = ["darkred","salmon","mistyrose",'b']
    #mainPlot=makeMainPlot(df.copy(),[lsdNick,cocNick,emNick,herNick],["darkred","salmon","mistyrose",'b'],["LSD","Cocaine","Ecstasy/Molly","Heroin"])
    #gammas= kurtSkew(df.copy(),[lsdNick,cocNick,emNick],["darkred","salmon","mistyrose","b"])
    #addGammas(df.index.values, mainPlot,gammas)
    #invGammaFrom4Moments(df,[lsdNick,cocNick,emNick])
    #invGamma(df,names,colors)
    ndf=df.copy()
    for name in names:
        ndf = uNormalizeColumn(ndf,name)
        ndf[name + wcstr] = ndf[name + wcstr].fillna(0)
        ndf[name + sestr] = ndf[name + sestr].fillna(3000.)
    cullenFrey(ndf,names,colors)
    #addDistPlot(df.copy(),names,colors,myLogNormal,(['-inf',0.],['inf','inf']))
if __name__ == '__main__':
    main()