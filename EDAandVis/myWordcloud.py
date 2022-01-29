from datetime import datetime, time
import time
from selenium.webdriver.chrome.service import Service
import numpy as np
import pandas as pd
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

import matplotlib.pyplot as plt

df = pd.read_csv(r'../Crawling/PriemerInj/all_inj.csv')
print(df.to_string())

text = df['Reason'].unique()
print(df['Reason'].values)
text = "/".join(inj for inj in df.Reason)

#stopwords = set(STOPWORDS)
stopwords = (["Injury", "Suspended", "Lower"])

#####with image frame:
# mask = np.array(Image.open("123.jpg"))
# image_colors = ImageColorGenerator(mask)
# plt.figure(figsize=[10,10])
# Mwordcloud = WordCloud(stopwords=stopwords, background_color="white", collocations=False,  mask=mask).generate(str(text))
# plt.imshow(Mwordcloud.recolor(color_func=image_colors), interpolation="bilinear")

wordcloud = WordCloud(stopwords=stopwords, collocations=False).generate(str(text))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
