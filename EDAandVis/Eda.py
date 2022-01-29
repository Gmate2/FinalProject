import matplotlib
import pandas as pd
from datetime import datetime, time
import time
from selenium.webdriver.chrome.service import Service
import numpy as np
import pyautogui
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Chrome
import os
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, linear_model, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn import metrics
import statsmodels.api as sm
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#########plot with Age effect#############
df = pd.read_csv(r'../LogisticRegModel/firstDraftAllData.csv')

sns.pairplot(df,corner=True,
             vars = ['inj22','Touches_x','Tackeled_x','Minutes / Div By 90_x','Number of Inj','Fouls Drawn_x',],
             hue = 'age', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
             size = 3)# Title
plt.suptitle('Pair Plot Age influence',
             size = 28)

plt.show()
#########plot with Position effect#############
df = pd.read_csv(r'../DataCleaning/fullDF/16-1FirstFULLclnDF.csv')
df.loc[(df["Position_DF"] == 1), 'Position'] = 'DF'
df.loc[(df["Position_DMF"] == 1), 'Position'] = 'DMF'
df.loc[(df["Position_MF"] == 1), 'Position'] = 'MF'
df.loc[(df["Position_AMF"] == 1), 'Position'] = 'AMF'
df.loc[(df["Position_FW"] == 1), 'Position'] = 'FW'
sns.pairplot(df,corner=True,
             vars = ['age','inj22','Career INJ days','Tackeled_y','Fouls Drawn_x','Number of Inj'],
             hue = 'Position', diag_kind = 'kde',
             plot_kws = {'alpha': 0.6, 's': 70, 'edgecolor': 'k'},
             size = 3)# Title
plt.suptitle('Pair Plot Position influence',
             size = 28)
plt.show()

# df = pd.read_csv(r'25-1Norm2.csv')
# grid = sns.PairGrid(data= df[df['CrsINJURY'] == 1],
#                     vars = ['inj22', 'Minutes / Div By 90_x',
#                     'Number of Inj'], size = 4)
# grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred',
#                      edgecolor = 'k')# Map a density plot to the lower triangle
# grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

# df = pd.read_csv(r'25-1Norm2.csv',sep=',')
# my_df = df[df["CrsINJURY"] == 1]
# obj = df.groupby('Team')['Christmas-played-games'].mean()
# obj1 = df.groupby('Team').count()
# df = pd.read_csv(r'PlotInj.csv')
# df["Team"]= LabelEncoder().fit_transform(df["Team"])
# sns.pairplot(df,corner=True,
#              vars = ['Team','Inj Players','Christmas-played-games','Number of Fit Players'], diag_kind = 'kde',
#              plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
#              size = 3)# Title
# plt.suptitle('Pair Plot Age influence',
#              size = 28)

#df = df.iloc[:, [1,2,5,9,14,15,16,19,20,21,22,23,24,27,28,29,31,32,35,39,40,41,43,45,51,53,54,55,57,58,60,61,62,63,64,66,67,69,72,
      #      80,81,82,84,87,88,89,90,91,93,100,105,106,107,108,109,110,111,113,114,116,117,118,123,124,125]]
#print(df.loc[df['CrsINJURY'] == 1].to_string())
#sns.set_theme(style="ticks")
#df1 = sns.load_dataset('25-1Norm2')
#sns.pairplot(df)
