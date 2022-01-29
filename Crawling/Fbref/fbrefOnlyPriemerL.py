###fbref Priemer League ONLY Data###
###############called by """"""GetFbrefData"""""################
import numpy as np
import pandas
import pandas as pd
import selenium
from numpy.random import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver import Chrome
import time
import urllib
from urllib.parse import unquote
from urllib.parse import urlparse, parse_qs
import pyautogui

def PrimerLeagueData (driver, url, teamName):

###go to website
    driver.get(url)
#in case of ads
    randomInt = np.random.randint(6)
    driver.implicitly_wait(5+randomInt)
    pyautogui.press('enter')
    time.sleep(3)
################check name!
    season = driver.find_element(By.XPATH, '//*[@id="meta"]/div[2]/h1/span[1]')
    name = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/th/a')
    nationalaty = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[1]')
    position = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[2]')
    age = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[3]')
    matchesPlayed = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[4]')
    starts = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[5]')
    totalMinutes = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[6]')
    minDivBy90 = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[7]')
    goals = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[8]')
    asissts = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[9]')
    yellows = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[13]')
    reds = driver.find_elements(By.XPATH, '//table[@id="stats_standard_11160"]/tbody/tr/td[14]')

    # Creating the firs DF
    playersRes = []
    for i in range(len(name)):
        tempData = {'Name': name[i].text,
                    'Team': teamName,
                    'Season': season.text,
                    'Nationalaty': nationalaty[i].text,
                    'Position': position[i].text,
                    'age': age[i].text,
                    'Matches Played': matchesPlayed[i].text,
                    'Matches Start': starts[i].text,
                    'Total Minutes': totalMinutes[i].text,
                    'Minutes / Div By 90': minDivBy90[i].text,
                    'Goals': goals[i].text,
                    'Asissts': asissts[i].text,
                    'Yellows': yellows[i].text,
                    'Reds': reds[i].text}
        playersRes.append(tempData)

    df_data = pd.DataFrame(playersRes)
    ##others tables data##:
    # player shots data
    nameShots = driver.find_elements_by_xpath('//*[@id="stats_shooting_11160"]/tbody/tr/th/a')  # (.text)
    totalShots = driver.find_elements_by_xpath('//*[@id="stats_shooting_11160"]/tbody/tr/td[6]')

    # player passes data
    namePasses = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/th/a')  # (.text)
    passesAttempted = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/td[6]')
    passesTotalDistance = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/td[8]')
    passesShort = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/td[11]')
    passesMedium = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/td[14]')
    passesLong = driver.find_elements_by_xpath('//*[@id="stats_passing_11160"]/tbody/tr/td[17]')

    # player ypasses type data
    namePassesType = driver.find_elements_by_xpath('//*[@id="stats_passing_types_11160"]/tbody/tr/th/a')  # (.text)
    passesCroses = driver.find_elements_by_xpath('//*[@id="stats_passing_types_11160"]/tbody/tr/td[12]')
    passsesHead = driver.find_elements_by_xpath('//*[@id="stats_passing_types_11160"]/tbody/tr/td[22]')

    # player defencive data
    nameDefence = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/th/a')  # (.text)
    tackeldRecived = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[5]')
    tackeldDribbelsRecived = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[10]')
    pressureApplying = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[14]')
    blocks = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[20]')
    blocksShots = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[21]')
    clearence = driver.find_elements_by_xpath('//*[@id="stats_defense_11160"]/tbody/tr/td[26]')

    # player Possession data
    namePossesion = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/th/a')  # (.text)
    touches = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[5]')
    touchesDefencivePenaltyArea = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[6]')
    touchesDefencive3RD = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[7]')
    touchesMid3RD = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[8]')
    touchesAttacking3RD = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[9]')
    touchesAttackingPenaltyArea = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[10]')
    touchesLiveGame = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[11]')
    dribbelsSucces = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[12]')
    dribbelsAttempted = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[13]')
    carries = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[17]')
    carriesToDistance = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[18]')
    carriesProggresiveDistance = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[19]')
    carriesProggresiveFoword = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[20]')
    carriesOpponent3RD = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[21]')
    carriesTackeled = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[24]')
    passesTarget = driver.find_elements_by_xpath('//*[@id="stats_possession_11160"]/tbody/tr/td[25]')

    # player Miscellaneous data
    nameMiscellaneous = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/th/a')  # (.text)
    foulsCommited = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[8]')
    foulsDrawn = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[9]')
    tackeleWon = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[13]')
    looseBallRecovered = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[17]')
    aerialDualsWon = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[18]')
    aerialDualsLost = driver.find_elements_by_xpath('//*[@id="stats_misc_11160"]/tbody/tr/td[19]')

    # creating the second DF
    playersRes1 = []
    for i in range(len(nameShots)):
        tempData1 = {'Name': nameShots[i].text,
                     'Total Shots': totalShots[i].text,
                     'Passes Attemted': passesAttempted[i].text,
                     'Passes Total Distance': passesTotalDistance[i].text,
                     'Short Passes': passesShort[i].text,
                     'Medium Passes': passesMedium[i].text,
                     'Long Passes': passesLong[i].text,
                     'Cross Passes': passesCroses[i].text,
                     'Head Pases': passsesHead[i].text,
                     'Tackeled': tackeldRecived[i].text,
                     'Tackeled Dribbels': tackeldDribbelsRecived[i].text,
                     'Pressure Applying': pressureApplying[i].text,
                     'Blocks': blocks[i].text,
                     'Shots Blocks': blocksShots[i].text,
                     'Clearence': clearence[i].text,
                     'Touches': touches[i].text,
                     'Touches Defencive Penalty Area': touchesDefencivePenaltyArea[i].text,
                     'Touches Defencive 3RD': touchesDefencive3RD[i].text,
                     'Touches Mid 3RD': touchesMid3RD[i].text,
                     'Touches Attacking 3RD': touchesAttacking3RD[i].text,
                     'Touches Attacking Penalty Area': touchesAttackingPenaltyArea[i].text,
                     'Touches Live Game': touchesLiveGame[i].text,
                     'Dribbels Succes': dribbelsSucces[i].text,
                     'Dribbels Attempted': dribbelsAttempted[i].text,
                     'Carries': carries[i].text,
                     'Carries To Distance': carriesToDistance[i].text,
                     'Carries Proggresive': carriesProggresiveDistance[i].text,
                     'Carries Proggresive Foword': carriesProggresiveFoword[i].text,
                     'Carries Tackeled': carriesTackeled[i].text,
                     'Reciving Rasses': passesTarget[i].text,
                     'Fouls Commited': foulsCommited[i].text,
                     'Fouls Drawn': foulsDrawn[i].text,
                     'Tackele Won': tackeleWon[i].text,
                     'looseBallRecovered': looseBallRecovered[i].text,
                     'Aerial Duals Won': aerialDualsWon[i].text,
                     'Aerial Duals Lost': aerialDualsLost[i].text,
                     }
        playersRes1.append(tempData1)

    df_data2 = pd.DataFrame(playersRes1)

    # scraping finished we can close the browser
    driver.quit()
    # marging the two data frames
    df_team = df_data.merge(df_data2, on='Name', how='left')

    return df_team
