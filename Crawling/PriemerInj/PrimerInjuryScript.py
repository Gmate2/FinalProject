###Daily Inj
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.chrome.service import Service

#chrome web driver at my! computer
def DailyInj():
    ser = Service("C:\Program Files (x86)\chromedriver.exe")
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("incognito")
    options.add_argument("--disable-extensions")
    DRIVER = webdriver.Chrome(service=ser, options=options)
    #inj link
    url = 'https://www.premierinjuries.com/injury-table.php'
    DRIVER.get(url)
    team = DRIVER.find_elements(By.XPATH, '//html/body/div[3]/div/section[4]/div/div/div/div[3]/div/div/div/table/tbody/tr/th/div/div[1]')

    for i in range(len(team)):
        DRIVER.implicitly_wait(2)
        team[i].click()
        time.sleep(2)

##stats var
    name = DRIVER.find_elements(By.XPATH,
                                     '//html/body/div[3]/div/section[4]/div/div/div/div[3]/div/div/div/table/tbody/tr/td[1]')

    absType = DRIVER.find_elements(By.XPATH,
                                      '//html/body/div[3]/div/section[4]/div/div/div/div[3]/div/div/div/table/tbody/tr/td[2]')

    potRe = DRIVER.find_elements(By.XPATH,
                                        '//html/body/div[3]/div/section[4]/div/div/div/div[3]/div/div/div/table/tbody/tr/td[4]')
    sta = DRIVER.find_elements(By.XPATH,
                                      '//html/body/div[3]/div/section[4]/div/div/div/div[3]/div/div/div/table/tbody/tr/td[6]')

    #get the corect team name for player
    idx = 0
    teamInfo = []
    for j in range(len(name)):

        if j >= 1:
            if name[j].text == "PLAYER":
                idx += 1
            else:
                tempLinks1 = {'Name': name[j].text,
                              'Team': team[idx].text,
                              'Reason': absType[j].text,
                              'Potantial Return': potRe[j].text,
                              'Status': sta[j].text,
                              'Date' : '04/01/2022'}
                teamInfo.append(tempLinks1)
    df_temp = pd.DataFrame(teamInfo)
    DRIVER.quit()

###save at local computer
    df_temp.to_csv('INJXX-01.csv', index=False)

    return df_temp

# old one:
    #team = DRIVER.find_elements(By.XPATH, '//table[@class="injury-table injury-table-full"]/tbody/tr/th')
