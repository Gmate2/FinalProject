###PART One - GATTHERING DATA###
import selenium
import pyautogui
import numpy as np
import pandas as pd
from numpy.random import random
from selenium import webdriver
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Chrome
from fbrefOnlyPriemerL import PrimerLeagueData
from fbrefGetLinks import fbrefGetLinks
from fbrefAllComp import AllCompData



# ################get Fbref Data: if only priemer legue stas wanted argg = False###############
# # fbrefScraper import GetFbrefData
# #df_temp = GetFbrefData (False)
# #print(df_temp.to_string())


# ############################F b R e f      s c r a p i n g################################################################
def GetFbrefData (allComp):
#open the wanted web page
#chrome web driver at my! computer
    ser = Service("C:\Program Files (x86)\chromedriver.exe")
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    DRIVER = webdriver.Chrome(service=ser, options=options)
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"

    links = fbrefGetLinks(DRIVER, url)



    df_team = pd.DataFrame()

    for i in range(len(links['Name'])):
        DRIVER = webdriver.Chrome(service=ser, options=options)
        teamName = links['Name'][i]
        url = links['link'][i]
        randomInt = np.random.randint(5)
        time.sleep(15+randomInt)
            ###for Priemer Only###
        if allComp == False:
            df_teamTemp = PrimerLeagueData(DRIVER, url, teamName)
            ###for All comp###
        else:
            df_teamTemp = AllCompData(DRIVER, url, teamName)
        frames = [df_team, df_teamTemp]
        df_team = pd.concat(frames)
        del df_teamTemp

###Saving at local Computer
    df_team.to_csv('fBrefAllComp='+str(allComp)+'.csv', index=False)