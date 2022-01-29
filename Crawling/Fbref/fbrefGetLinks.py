###SECOND FUNC - GATTHERING Team Links###
import numpy as np
import pandas
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Chrome
import time
from fbrefAllComp import AllCompData
from fbrefOnlyPriemerL import PrimerLeagueData
import pyautogui
###First we'll get the link for each team in the Priemer League
def fbrefGetLinks(driver, url):

#go to
    driver.get(url)
#in case of ads
    randomInt = np.random.randint(6)
    driver.implicitly_wait(5 + randomInt)
    pyautogui.press('enter')
#the priemer legue teams
    teams = ['Manchester City', 'Liverpool', 'Chelsea', 'Arsenal', 'West Ham', 'Manchester Utd', 'Tottenham', 'Wolves',
             'Leicester City', 'Aston Villa', 'Crystal Palace', 'Brentford', 'Brighton', 'Everton', 'Southampton',
             'Leeds United', 'Watford', 'Burnley', 'Newcastle Utd', 'Norwich City']
#get the links for relevant team
    team = driver.find_elements(By.XPATH, '//div[@id="all_results111601"]//table[@id="results111601_overall"]/tbody/tr/td/a')
    links = []
#create th DF
    for i in range(len(team)):
        if team[i].text in teams:
            tempLinks1 = {'Name': team[i].text,
                          'link': team[i].get_attribute("href"),
                          }

            links.append(tempLinks1)
        else:
            continue
    df_links = pd.DataFrame(links)
    driver.quit()
    time.sleep(2)
    #return the DF links
    return df_links
