import pandas as pd
import numpy as np
import seaborn as  sns
import matplotlib.pyplot as plt
#from sklearn import preprocesssing
#from uncertainties import ufloat

wcstr = "Weighted Count"
sestr = "Count SE"

def makeDrugFrame():


def makeDrugFrame(drugName, nickname, filepath):
    agerange = range(10, 66)
    tried_df = pd.read_csv(filepath)
    tried_df = tried_df[[drugName, wcstr, sestr]]
    tried_df = tried_df[tried_df[drugName].apply(lambda x: x.isnumeric() and int(x) in agerange)]
    pd.to_numeric(tried_df[drugName])
    tried_df.set_index(drugName, inplace=True)
    tried_df.sort_index(inplace=True)
    tried_df.rename(columns=lambda x: nickname+' '+ x, inplace=True)
    return tried_df

def makeDistPlot(df,nickname,color):
    plot = df[nickname+' '+wcstr].plot(kind='bar',yerr=df[nickname+' '+sestr])
    return plot
def makeMainPlot(df,names):
    x=1
    for name in names:
        for i in df[name+' '+wcstr]:
            print(str(x)+':'+str(i))
            x=x+1


def main():
    str_lsd = "AGE WHEN FIRST USED LSD"
    lsd_path = r"C:\Users\Francesco Vassalli\Downloads\LSDage.csv"
    lsdNick = 'LSD: '
    df=makeDrugFrame(str_lsd, lsdNick, lsd_path)
    df.index.names = ['Age']
    str_coc = "AGE WHEN FIRST USED COCAINE"
    coc_path = r"C:\Users\Francesco Vassalli\Downloads\COCage.csv"
    cocNick = 'COC: '
    str_em = "AGE WHEN FIRST USED ECSTASY OR MOLLY"
    em_path = r"C:\Users\Francesco Vassalli\Downloads\ECSTMOAGE.csv"
    emNick = 'E/M: '
    df=df.join([makeDrugFrame(str_coc, cocNick, coc_path),makeDrugFrame(str_em,emNick,em_path)])
    print(df)
    makeMainPlot(df,[lsdNick,cocNick,emNick])
    #lsd_dist_plot = makeDistPlot(df,lsdNick,'darkred')
    #coc_dist_plot = makeDistPlot(df,cocNick,'salmon')
    #em_dist_plot = makeDistPlot(df,emNick,'mistyrose')
    #plt.show()

if __name__ == '__main__':
    main()