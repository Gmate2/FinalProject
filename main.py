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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


df = pd.read_csv(r'DataCleaning/fullDF/16-1FirstFULLclnDF.csv', sep=',')
print(df.info(verbose = False))