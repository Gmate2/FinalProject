import pandas as pd
from datetime import datetime, time
import time
from selenium.webdriver.chrome.service import Service
import numpy as np
import pyautogui
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
#print(df['Name'].isin(df1['Name']).value_counts())


# df1 = pd.read_csv('LASTINJ12-1.csv')
# result = pd.merge(df, df1, how="left",on=["Name"])
# result = result.fillna(0)
# result["Team"]= LabelEncoder().fit_transform(result["Team"])
# result = result.set_index('Name',drop=True)

#print(result.to_csv())
#result.to_csv('check.csv', index=True)
#print(result.to_string())
#print(df_playersTemp.to_string())

df = pd.read_csv(r'PriemerInjCln/ChrisInjFirstClean.csv', sep=',')
df["Team"] = df.Team.str.split("\nT",expand=False).str[0]
#print(df["Name"].value_counts())
df = df.groupby(['Name',"Team","Reason"], as_index = False).sum()

df = df.drop(df[df.Reason == 'Illness'].index)
df = df.drop(df[df.Reason == 'Suspended'].index)
print(df.to_string())
#
# CheckDF['Date'] = pd.to_datetime(CheckDF['Date'], dayfirst=True,errors='ignore')
#
# df = pd.get_dummies(CheckDF, columns=['Date'])
# df = df.sort_values(by=['Name'])
# #df["Team"] = df["Team"].str.split('\\')
# #print(df["Name"].value_counts().to_string())
# df = df[["Name","Team", "Reason",'Date_2022-12-26 00:00:00', 'Date_2022-12-27 00:00:00', 'Date_2022-12-28 00:00:00', 'Date_2022-12-29 00:00:00', 'Date_2022-12-30 00:00:00',
#          "Date_2022-12-31 00:00:00","Date_2022-01-01 00:00:00","Date_2022-01-02 00:00:00","Date_2022-01-03 00:00:00","Date_2022-01-04 00:00:00"]]
#
# df = df.groupby(['Name',"Team","Reason"], as_index = False).sum()
# #df = df.loc[(df['Date_2022-12-26 00:00:00'] == 1) & (df["Date_2022-01-03 00:00:00"]==1)]
# df = df.drop(354,axis=0)
# df = df.drop(447,axis=0)
# df = df.drop(473,axis=0)
# df = df.drop(353,axis=0)
# df = df.drop(379,axis=0)
# df = df.drop(325,axis=0)
# df = df.drop(61,axis=0)
# df = df.drop(62,axis=0)
# df = df.drop(162,axis=0)
# df = df.drop(518,axis=0)
# df = df.drop(519,axis=0)
# df = df.drop(144,axis=0)
# df = df.drop(248,axis=0)
# df = df.drop(15,axis=0)
# df = df.drop(112,axis=0)
# df = df.drop(359,axis=0)
# df = df.drop(426,axis=0)
# df = df.drop(146,axis=0)
# df = df.drop(509,axis=0)
# df = df.drop(5,axis=0)
# df = df.drop(184,axis=0)
# df = df.drop(433,axis=0)

df = pd.read_csv(r'PriemerInjCln/15-1InjWatch.csv', sep=',')
df1 = pd.read_csv(r'PriemerInjCln/PlayersToDrop.csv', sep=',')
df = df.drop(df[df['Name'].isin(df1['Name'])].index)

#names = CheckDF['Name'].unique()
# newLst = []
# for i in range(len(names)):
#     df['Number of Days'] = CheckDF.loc[(CheckDF['Name'] == names[i])].count()
#     df['Reason'] = (CheckDF.loc[CheckDF['Name'] == names[i], 'Reason'].value_counts().idxmax())

#df.to_csv('ChrisInjFirstClean.csv', index=False)
# names = df['Name'].unique()
# newLst = []
# for i in range(len(names)):
#     tempPar = {
#         'Name': names[i],
#         'Team': (CheckDF.loc[CheckDF['Name'] == names[i], 'Team'].value_counts().idxmax()),
#         'Number of Inj': CheckDF.loc[
#             (CheckDF['Name'] == names[i]) & (CheckDF["Type of Absence"] != 'Suspended'), 'Type of Absence'].count(),
#         'inj22': CheckDF.loc[CheckDF['Name'] == names[i], 'Total inj days 22'].sum(),
#         'inj days 20-21': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 20-21'].sum(),
#         'inj days 19-20': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 19-20'].sum(),
#         'inj days 18-19': CheckDF.loc[CheckDF['Name'] == names[i], 'Total Inj days 18-19'].sum(),
#         'Sus days 22': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 22'].sum(),
#         'sus days 20-21': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 20-21'].sum(),
#         'sus days 19-20': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 19-20'].sum(),
#         'sus days 18-19': CheckDF.loc[CheckDF['Name'] == names[i], 'Suspension days 18-19'].sum(),
#         'Career Suspension days': CheckDF.loc[CheckDF['Name'] == names[i], 'Career Suspension days'].sum(),
#         'Career INJ days': CheckDF.loc[CheckDF['Name'] == names[i], 'Career INJ days'].sum(),
#
#     }
#     newLst.append(tempPar)
#
# df_playersTemp = pd.DataFrame(newLst, index=None)
#counts = df['Name'].value_counts(dropna=False)
#df = df[df['Name'].map(counts)>9]
#print(df)
#print(df['Name'].value_counts())

# df = pd.read_csv(r'INJWatchAll.csv')
# df1 = pd.read_csv(r'PlayersToDrop.csv')
# df=df[["Name","Reason"]]
# #print(df.to_string())
# df2 = df.groupby( [ "Name", "Reason"] ).count().reset_index()
# df2 = df2[df2['Name'].isin(df1['Name'])]