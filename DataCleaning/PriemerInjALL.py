
from datetime import datetime, time
import time
import numpy as np
import pandas as pd


######inj Daily List
#df_temp = DailyInj()
#print(df_temp.to_string())




df = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ26-12.csv')
df1 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ27-12.csv')
df2 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ28-12.csv')
df3 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ29-12.csv')
df4 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ30-12.csv')
df5 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ31-12.csv')
df6 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ01-01.csv')
df7 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ02-01.csv')
df8 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ03-01.csv')
df9 = pd.read_csv(r'../Crawling/PriemerInj/InjWatch/INJ04-01.csv')
df = df.append([df1,df2,df3,df4,df5,df6,df7,df8,df9] ,sort=True)
df = df.sort_values(by=['Name','Date','Team'])
print(df.to_string())
df.to_csv('INJWatchAll.csv', index=False)

#
# ###Draft
# # df = pd.read_csv(r'INJ26-12.csv')
# # df.loc[(df["Reason"] != 'Suspended') & (df["Reason"] != 'Illness'), 'InjDays'] = 1
# # df1 = pd.read_csv(r'INJ27-12.csv')
# # df1['New'] = df1['Name'].isin(df['Name'])
# # df1.loc[(df1["Reason"] != 'Suspended') & (df1["Reason"] != 'Illness'), 'InjDays'] = 1
#
# #print(df1.to_string())
# # df2 = pd.read_csv(r'INJ28-12.csv')
# # df2['New'] = df2['Name'].isin(df1['Name'])
# # df2.loc[(df2["Reason"] != 'Suspended') & (df2["Reason"] != 'Illness'), 'InjDays'] = 1
# # #print(df2.to_string())
# # df3 = pd.read_csv(r'INJ29-12.csv')
# # df3.loc[(df3["Reason"] != 'Suspended') & (df3["Reason"] != 'Illness'), 'InjDays'] = 1
# # df4 = pd.read_csv(r'INJ30-12.csv')
# # df4.loc[(df4["Reason"] != 'Suspended') & (df4["Reason"] != 'Illness'), 'InjDays'] = 1
# # df5 = pd.read_csv(r'INJ31-12.csv')
# # df5.loc[(df5["Reason"] != 'Suspended') & (df5["Reason"] != 'Illness'), 'InjDays'] = 1
# # df6 = pd.read_csv(r'INJ01-01.csv')
# # df6.loc[(df6["Reason"] != 'Suspended') & (df6["Reason"] != 'Illness'), 'InjDays'] = 1
# # df7 = pd.read_csv(r'INJ02-01.csv')
# # df7.loc[(df7["Reason"] != 'Suspended') & (df7["Reason"] != 'Illness'), 'InjDays'] = 1
# # df = df.append([df1,df2,df3,df4,df5,df6,df7] ,sort=True)
# # df = df.sort_values(by=['Name','Date','Team'])
# # print(df['Reason'].value_counts().to_string())
# # print(df['New'].value_counts().to_string())
# #print(df.to_string())
#
# #df.to_csv('INJ31-12.csv', index=False)
#
# ########INJORY##################
# # all_inj_data = pd.DataFrame()
# # files = [file for file in os.listdir(r'./PriemerInj/InjWatch/')]
# # for file in files:
# #     df = pd.read_csv(r'./PriemerInj/InjWatch/'+file)
# #     all_inj_data = pd.concat([all_inj_data,df])
# # all_inj_data.to_csv("all_inj.csv", index=False)
#
# df = pd.read_csv(r'all_inj.csv')
# df['Date'] = pd.to_datetime(df['Date'], dayfirst=True,errors='ignore')
# df = df.sort_values(by=['Name','Date','Team'])
# #in case of na rows:
# #nan_df = df[df.isna().any(axis=1)]
# #df = df.dropna(how='all')
# #convert to int:
# #df = pd.to_numeric(df)
# #print(df['Reason'].value_counts())
# #df1 = pd.read_csv(r'soccerway/2-1-soccerWayINJ.csv')
# #print(df1['Type of Absence'].value_counts()[:15])
#df = pd.read_csv(r'Fbref/AllComp.csv')
#print(df.to_string())

