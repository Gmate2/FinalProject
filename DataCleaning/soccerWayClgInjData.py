###PART ONE - GATTHERING DATA###
from datetime import datetime
from selenium.webdriver.chrome.service import Service
import numpy as np
import pandas as pd
from selenium.webdriver.chrome.options import Options

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
#DRIVER = webdriver.Chrome(service=ser, options=options)



#df = pd.read_csv(r'SoccerWay27-11INJ.csv')

#Checkdf[['First','Last']] = df.Name.str.split(".",expand=True)
#df[['First','Last']]= df.Name.str.rsplit(pat='.',n=1,expand=True)
#df[['First1','Last1']]= df.First.str.rsplit(pat=' ',n=1,expand=True)
#for i in range(len(df['First'])):
# if df['First'][i] == df['First1'][i]:
# 2df.loc[lambda df: df['First1']] = 0

  #      df[] = df.New.str.split(pat=' ',n=-1,expand=True)
    #finally:
     #   continue

#df["name",'new']= df["Name"].str.split(pat='.', n = 0, expand = True)
#df[['First','Last']] = df.Name.str.split(".",expand=True)
#df[['TeamG']] = df.Team.str.split("\n",expand=True)
#print(CheckDF.to_string())

#CheckDF[['First','Last']] = CheckDF.Name.str.split(expand=True)
#print(df.to_string())
# df = pd.read_csv(r'SoccerWay27-11INJ.csv')
# df2 = pd.read_csv(r'INJ26-12.csv')
# df.fillna('26/12/21', inplace=True)
# #print(df.to_string())
# #mising = df.merge(df2, on='Name', how='left')
#
# print(df[df['End Date'] == '26/12/21'])
#for i in range(len(mising)):


#print(mising.to_string())
#         url = df_AllPlayers["link"][i]
#     randomInt = np.random.randint(7)
#     DRIVER.get(url)
#     try:
#         DRIVER.implicitly_wait(4 + randomInt)
#
#         endDate = DRIVER.find_element(By.XPATH,
#                                         '//div[@id="page_player_1_block_player_sidelined_9-wrapper"]/div/div/table/tbody/tr/td[4]')
#
#      ####get injury & susspention data for each player:
#         for j in range(len(NumberOfInj)-1):
#
#                           'End Date': endDate[j].text
#
#
#
#      finally:
#              DRIVER.quit()
# df_playersTemp = pd.DataFrame(playerInj)
# #frames = [df_playersInj, df_playersTemp]
# #df_playersInj = pd.concat(frames)
# df_playersTemp.to_csv('NsoccerWayINJ'+str(i), index=False)
# #del df_playersTemp
#
# # print(df_playersInj.to_string())

CheckDF = pd.read_csv(r'soccerWay/2-1-soccerWayINJ.csv')
#print(CheckDF.head())

###############################################################################
###aplly today as 'end date' for inj players
CheckDF['End Date'].fillna('03/01/2022',inplace=True)

###convert inj dates to 'datetime' format
CheckDF['Start Date'] = pd.to_datetime(CheckDF['Start Date'], dayfirst=True,errors='ignore')
CheckDF['End Date'] = pd.to_datetime(CheckDF['End Date'], dayfirst=True,errors='ignore')

#correct mistaken year at 'end date'
CheckDF.loc[(CheckDF["Start Date"] > CheckDF['End Date']), 'End Date'] = CheckDF['End Date'] + pd.DateOffset(years=1)
###calculate the abssence days and convert it to int
CheckDF['Total Days'] = (CheckDF['End Date'] - CheckDF['Start Date'])
CheckDF['Total Days'] = CheckDF['Total Days'].dt.days

###set 22 season params
start = datetime(2021,8,1)
end = datetime(2021,12,26)

CheckDF.loc[(CheckDF["Start Date"]  > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] != 'Suspended'),
            'Total inj days 22'] = CheckDF['Total Days']

CheckDF.loc[(CheckDF["Start Date"]  > start) & (CheckDF["Type of Absence"] == 'Suspended'), 'Suspension days 22'] = CheckDF['Total Days']

###set 20-21 season params
start = datetime(2020,8,1)
end = datetime(2021,8,1)

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] != 'Suspended'),
            'Total Inj days 20-21'] = CheckDF['Total Days']

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] == 'Suspended'),
            'Suspension days 20-21'] = CheckDF['Total Days']

###set 19-20 season params
start = datetime(2019,8,1)
end = datetime(2020,8,1)

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] != 'Suspended'),
            'Total Inj days 19-20'] = CheckDF['Total Days']

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] == 'Suspended'),
            'Suspension days 19-20'] = CheckDF['Total Days']

###set 18-19 season params
start = datetime(2018,8,1)
end = datetime(2019,8,1)

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] < end) & (CheckDF["Type of Absence"] != 'Suspended'),
            'Total Inj days 18-19'] = CheckDF['Total Days']

CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] <= end) & (CheckDF["Type of Absence"] == 'Suspended'),
            'Suspension days 18-19'] = CheckDF['Total Days']


# CheckDF.loc[(CheckDF["Start Date"] > start) & (CheckDF['End Date'] > end) & (CheckDF["Type of Absence"] != 'Suspended'),
#             'Total inj days 22'] =(CheckDF['Total Days'] - 365)


###set career INJ days
CheckDF.loc[CheckDF["Type of Absence"] != 'Suspended', 'Career INJ days'] = CheckDF['Total Days']
###set career suspension days
CheckDF.loc[CheckDF["Type of Absence"] == 'Suspended', 'Career Suspension days'] = CheckDF['Total Days']

#print(CheckDF.to_string())
names = CheckDF['Name'].unique()
newLst = []
for i in range(len(names)):
#get the most frequent injory-> put 'illnes' & 'virus' last place
    try:
        Inj = (CheckDF.loc[(CheckDF['Name'] == names[i]) & (CheckDF["Type of Absence"] != 'Suspended'),
                                                               'Type of Absence'].value_counts().sort_index(ascending=True).sort_values(ascending=False).idxmax())
        if Inj == 'Illness':
            try:
                Inj = (CheckDF.loc[(CheckDF['Name'] == names[i]) & (CheckDF["Type of Absence"] != 'Suspended') & (CheckDF["Type of Absence"] != 'Illness'),
                                  'Type of Absence'].value_counts().sort_index(ascending=True).sort_values(ascending=False).idxmax())
                if Inj == 'Virus':
                    try:
                        Inj = (CheckDF.loc[(CheckDF['Name'] == names[i]) & (CheckDF["Type of Absence"] != 'Suspended') &
                               (CheckDF["Type of Absence"] != 'Illness') & (CheckDF["Type of Absence"] != 'Virus'),
                               'Type of Absence'].value_counts().sort_index(ascending=True).sort_values(ascending=False).idxmax())
                    except:
                            Inj = 'Virus'
            except:
                Inj = 'Illness'

    except:
        Inj = np.nan
#create the new params
    tempPar = {
                'Name': names[i],
                'First Name' : (CheckDF.loc[CheckDF['Name'] == names[i], 'First Name'].value_counts().idxmax()),
                'Last Name' : (CheckDF.loc[CheckDF['Name'] == names[i], 'Last Name'].value_counts().idxmax()),
                'Team' : (CheckDF.loc[CheckDF['Name'] == names[i], 'Team'].value_counts().idxmax()),
                'Number of Inj': CheckDF.loc[(CheckDF['Name'] == names[i]) & (CheckDF["Type of Absence"] != 'Suspended') , 'Type of Absence'].count(),
                'inj22': CheckDF.loc[CheckDF['Name'] == names[i], 'Total inj days 22'].sum(),
                'Freq Inj': Inj,
                'inj days 20-21': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 20-21'].sum(),
                'inj days 19-20': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 19-20'].sum(),
                'inj days 18-19': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 18-19'].sum(),
                'Sus days 22': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 22'].sum(),
                'sus days 20-21': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 20-21'].sum(),
                'sus days 19-20': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 19-20'].sum(),
                'sus days 18-19': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 18-19'].sum(),
                'Career Suspension days': CheckDF.loc[CheckDF['Name'] == names[i], 'Career Suspension days'].sum(),
                'Career INJ days': CheckDF.loc[CheckDF['Name'] == names[i], 'Career INJ days'].sum(),

                }
    newLst.append(tempPar)

df_playersTemp = pd.DataFrame(newLst,index=None)

###get dummies for inj
inj = CheckDF['Type of Absence'].value_counts()[lambda x: x > 11].index
df = pd.get_dummies(pd.Categorical(CheckDF['Type of Absence'], categories=inj))
df["Name"] = CheckDF['Name']
df = df.groupby(['Name'], as_index = False).sum()
result = pd.merge(df_playersTemp, df, how="left",on=["Name"])


result.to_csv('Allinj11-1.csv', index=False)
#df_playersTemp.to_csv('2-1-ALLinj', index=False)
#print(CheckDF.loc[CheckDF['Name'] == 'M. Klich'].to_string())
print(df_playersTemp.to_string())

#CheckDF[['First','Last']] = CheckDF.Name.str.split(".",expand=True)
#CheckDF[['First','Last']]= (CheckDF["Name"].str.split(pat='.', n = -1, expand = True)).convert_dtypes()
#CheckDF['First'] = CheckDF['First'].convert_dtypes()
#CheckDF['Last'] = CheckDF['Last'].convert_dtypes()
#CheckDF[['first1','last1']]= CheckDF["Name"].str.split(pat=' ', n = 0, expand = True)
#print(CheckDF.head())
#print(CheckDF.head().to_string())
# df = CheckDF.copy()

###for Ploting:########
df = pd.read_csv(r'16-1FirstFULLclnDF.csv')
my_df = df.groupby('Team')['CrsINJURY'].apply(lambda x: (x==1).sum()).reset_index(name='Inj Players')

my_df1 = df.groupby('Team')['Christmas-played-games'].mean()
my_df3 = df.groupby('Team')['Name'].count().reset_index(name='Number of Fit Players')
my_df2 = my_df.merge(my_df1, on='Team', how='left')
dfAll = my_df2.merge(my_df3, on='Team', how='left')
dfAll.to_csv('PlotInj.csv', index=False)
print(dfAll.to_string())


