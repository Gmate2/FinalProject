###PART ONE - GATTHERING DATA###
from selenium.webdriver.chrome.service import Service
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

###avoid selenium block####
#options.add_experimental_option("excludeSwitches",["enable-automation"])
#options.add_experimental_option('useAutomationExtension', False)
#stealth(DRIVER,languages =["en-US","en"],vendor="Google Inc.",platform= "Win32",
# webgl_vendor="Intel Inc.",renderer="Intel Iris OpenGL Engine",fix_hairline=True)


#chrome web driver at my! computer

ser = Service("C:\Program Files (x86)\chromedriver.exe")
options = Options()
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("incognito")
options.add_argument("--disable-extensions")
DRIVER = webdriver.Chrome(service=ser, options=options)



###Soccer Way Inj Data#####
#soccerway url
###part ONE####
url = 'https://int.soccerway.com/national/england/premier-league/20212022/regular-season/r63396/'
DRIVER.get(url)

#get each team link
team = DRIVER.find_elements(By.XPATH, '//*[@id="page_competition_1_block_competition_tables_13"]/div/form/table/tbody/tr/td[3]/a')
links = []

for i in range(len(team)):
    tempLinks1 = {'Name': team[i].text,
                  'link': team[i].get_attribute("href"),
                      }

    links.append(tempLinks1)

df_links = pd.DataFrame(links)
time.sleep(3)
DRIVER.quit()


###part TWO###
#get evrey player in each team link:

df_AllPlayers = pd.DataFrame()
linksPlayer = []
for i in range(len(df_links['link'])):
    DRIVER = webdriver.Chrome(service=ser, options=options)
    url = df_links['link'][i]
    DRIVER.get(url)
#avoid blocking
    randomInt = np.random.randint(6)
    DRIVER.implicitly_wait(5 + randomInt)
#get the names
    player = DRIVER.find_elements(By.XPATH,
                                  '//*[@id="page_team_1_block_team_squad_12-wrapper"]/div/div/div/table/tbody//tr/td/div/a')

    #get evrey player of the team link (-1 == minus coach!)
    for j in range(len(player)-1):

        tempLinks2 = {'Name': player[j].text,
                      'Team': df_links['Name'][i],
                      'link': player[j].get_attribute("href"),
                       }

        linksPlayer.append(tempLinks2)
        DRIVER.quit()
        time.sleep(2)

df_playersLinksTemp = pd.DataFrame(linksPlayer)
#frames = [df_AllPlayers,  df_playersLinksTemp]
#df_AllPlayers = pd.concat(frames)
#del df_playersLinksTemp


df_AllPlayers.to_csv(r'NsoccerWayPlayersLinks.csv', index=False)
print(df_AllPlayers.to_string())

###part THREE###
df_AllPlayers = pd.read_csv(r'soccerWayPlayersLinksClean.csv')
#get evrey player Inj history
playerInj = []
df_playersInj = pd.DataFrame()
for i in range(len(df_AllPlayers['Name'])):
     DRIVER = webdriver.Chrome(service=ser, options=options)
     url = df_AllPlayers["link"][i]
     randomInt = np.random.randint(7)
     DRIVER.get(url)
     try:
         DRIVER.implicitly_wait(4 + randomInt)

         firstName = DRIVER.find_elements(By.XPATH,
                                            '//*[@id="page_player_1_block_player_passport_3"]/div/div/div[1]/div/dl/dd[1]')

         lastName = DRIVER.find_elements(By.XPATH,
                                          '//*[@id="page_player_1_block_player_passport_3"]/div/div/div[1]/div/dl/dd[2]')

         NumberOfInj = DRIVER.find_elements(By.XPATH,'//div[@id="page_player_1_block_player_sidelined_9-wrapper"]/div/div/table/tbody/tr' )

         typeOfAbsence = DRIVER.find_elements(By.XPATH,
                                              '//div[@id="page_player_1_block_player_sidelined_9-wrapper"]/div/div/table/tbody/tr/td[2]')

         startDate = DRIVER.find_elements(By.XPATH,
                                          '//div[@id="page_player_1_block_player_sidelined_9-wrapper"]/div/div/table/tbody/tr/td[3]')

         endDate = DRIVER.find_elements(By.XPATH,
                                        '//div[@id="page_player_1_block_player_sidelined_9-wrapper"]/div/div/table/tbody/tr/td[4]')

     ####get injury & susspention data for each player:
         for j in range(len(NumberOfInj)-1):
             print(firstName[0].text)
             tempLinks3 = {'First Name': firstName[0].text,
                           'Last Name': lastName[0].text,
                           'Name': df_AllPlayers['Name'][i],
                           'Team': df_AllPlayers['Team'][i],
                           'Type of Absence': typeOfAbsence[j].text,
                           'Start Date': startDate[j].text,
                           'End Date': endDate[j].text
                           }
             playerInj.append(tempLinks3)

     finally:
             print(df_AllPlayers["link"][i])
             DRIVER.quit()
df_playersTemp = pd.DataFrame(playerInj)
#frames = [df_playersInj, df_playersTemp]
#df_playersInj = pd.concat(frames)
df_playersTemp.to_csv('NsoccerWayINJ', index=False)
#del df_playersTemp

# print(df_playersInj.to_string())


#############in case of Mising players:############################
#df_AllPlayers['Mis'] = df_AllPlayers['Name'].isin(df1['Name'])
#df_AllPlayers['MisNew'] = df_AllPlayers['Name'].isin(df['Name'])
#print(df_AllPlayers['Mis'] .value_counts())

#if (df_AllPlayers['MisNew'][i] == False) & (df_AllPlayers['Mis'][i] == True):

# tempdf = pd.read_csv(r'FinalMISSINGsoccerWayINJ.csv')
#
# newdf = df.append(tempdf, ignore_index=True)
# newdf = newdf.sort_values(by=['Team','Name'])
# newdf.to_csv('2-1-soccerWayINJ', index=False)
# print(newdf.to_string())
# #



