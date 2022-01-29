import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

##dealing with player who were 'Loan' and play in two different club:
#current season

# df = pd.read_csv(r'Fbref/AllComp.csv')
# df2 = df.select_dtypes(include=['object'])
# df2 = df2.drop(['Total Minutes','Season'], axis = 1)
# df_temp = df.groupby(['Name']).sum(numeric_only = True).reset_index()
# result = pd.merge(df2, df_temp, how="right",on=["Name"])
# df = result.drop_duplicates(['Name'], keep = 'last')
#last season

# df = pd.read_csv(r'Fbref/LastSeason.csv')
# df2 = df.select_dtypes(include=['object'])
# df2 = df2.drop(['Total Minutes','Season'], axis = 1)
# df_temp = df.groupby(['Name']).sum(numeric_only = True).reset_index()
# result = pd.merge(df2, df_temp, how="right",on=["Name"])
# df = result.drop_duplicates(['Name'], keep = 'last')
# df = df.drop(['age'], axis = 1)
# df.to_csv('LastSeason10-1.csv', index=False)
#df.to_csv('AllComp10-1.csv', index=False)
df = pd.read_csv(r'../Crawling/Fbref/AllComp10-1.csv')
df1 = pd.read_csv(r'../Crawling/Fbref/LastSeason10-1.csv')

df1.drop(columns=["Team","Nationalaty","Position"],inplace=True)
result = pd.merge(df, df1, how="left",on=["Name"])
#drop players that didn't play"
df = result.drop(result[result["Matches Played_x"] < 2].index)
#result.to_csv('temp13.csv', index=False)


# #Change categorical to numbers:
#df["Team"]= LabelEncoder().fit_transform(df["Team"])
#df["Nationalaty"]= LabelEncoder().fit_transform(df["Nationalaty"])
#df["Position"]= LabelEncoder().fit_transform(df["Position"])

# #change age to number
df["age"] = df.age.str.split("-",expand=False).str[0]
df["age"] = pd.to_numeric(df["age"])
df = df.fillna(0)

df['Name'] = df['Name'] .str.replace('-',' ')
df["first"] = np.nan
df["last"] = np.nan
df["third"] = np.nan
try:
 df["first"] = df["Name"].str.split(" ").str.get(0)
 df["last"] = df["Name"].str.split(" ").str.get(1)
 df["third"] = df["Name"].str.split(" ").str.get(2)
except:
    print('i')
###make position more uniqe and get dummies
df.loc[(df["Position"] == 'FW,DF') , 'Position'] = 'DMF'
df.loc[(df["Position"] == 'DF,FW') , 'Position'] = 'DMF'
df.loc[(df["Position"] == 'MF,DF') , 'Position'] = 'DMF'
df.loc[(df["Position"] == 'DF,MF') , 'Position'] = 'DMF'
df.loc[(df["Position"] == 'MF,FW') , 'Position'] = 'AMF'
df.loc[(df["Position"] == 'FW,MF') , 'Position'] = 'AMF'
df = pd.get_dummies(df, columns=['Position'])

##set natinalaty is english and not english
df.loc[df['Nationalaty'] != "eng ENG", 'Nationalaty'] = 0
df.loc[df['Nationalaty'] == "eng ENG", 'Nationalaty'] = 1

df.to_csv('Merge10-1.csv', index=False)


###dealing with inj data:
df = pd.read_csv(r'../Crawling/Fbref/Merge10-1.csv')
df1 = pd.read_csv(r'../Crawling/soccerWay/2-1-ALLinj.csv')
df1["Freq Inj"]= LabelEncoder().fit_transform(df1["Freq Inj"])
df1.drop(columns=["Team","Last Name","First Name"],inplace=True)
result = pd.merge(df, df1, how="right", on="Name")
df2 = result.dropna()
print(df2.to_string())

###Drop all goalkeepers
df = df.drop(df[df['Position_GK'] == 1].index)
df = df.drop(['Position_GK'], axis=1)

###Drop players Who bearly played
df = df.drop(df[df['Minutes / Div By 90_x'] < 2].index)

###Normalize per game:
df = df.drop(columns=['Matches Starts_x', 'Matches Starts_y'])
normToTimeX = ["Goals_x","Total Shots_x","Passes Total Distance_x","Asissts_x","Yellows_x","Reds_x","Passes Attemted_x","Short Passes_x","Medium Passes_x",
              "Long Passes_x","Cross Passes_x","Head Pases_x","Tackeled_x","Tackeled Dribbels_x","Pressure Applying_x","Blocks_x",
              "Shots Blocks_x","Clearence_x","Touches_x","Touches Defencive Penalty Area_x","Touches Defencive 3RD_x","Touches Defencive 3RD_x",
              "Touches Mid 3RD_x","Touches Attacking 3RD_x","Touches Attacking Penalty Area_x","Touches Live Game_x","Dribbels Succes_x",
              "Dribbels Succes_x","Dribbels Attempted_x","Carries_x","Carries To Distance_x","Carries Proggresive_x",
              "Carries Proggresive Foword_x","Carries Tackeled_x","Reciving Passes_x","Fouls Commited_x","Fouls Drawn_x",
              "Tackele Won_x","looseBallRecovered_x","Aerial Duals Won_x","Aerial Duals Lost_x"]

normToTimeY = ["Goals_y","Total Shots_y","Passes Total Distance_y","Asissts_y","Yellows_y","Reds_y","Passes Attemted_y","Short Passes_y","Medium Passes_y",
              "Long Passes_y","Cross Passes_y","Head Pases_y","Tackeled_y","Tackeled Dribbels_y","Pressure Applying_y","Blocks_y",
              "Shots Blocks_y","Clearence_y","Touches_y","Touches Defencive Penalty Area_y","Touches Defencive 3RD_y","Touches Defencive 3RD_y",
              "Touches Mid 3RD_y","Touches Attacking 3RD_y","Touches Attacking Penalty Area_y","Touches Live Game_y","Dribbels Succes_y",
              "Dribbels Succes_y","Dribbels Attempted_y","Carries_y","Carries To Distance_y","Carries Proggresive_y",
              "Carries Proggresive Foword_y","Carries Tackeled_y","Reciving Passes_y","Fouls Commited_y","Fouls Drawn_y",
              "Tackele Won_y","looseBallRecovered_y","Aerial Duals Won_y","Aerial Duals Lost_y"]

for col in normToTimeX:
    df[col] = df[col]/df['Minutes / Div By 90_x']
for col in normToTimeY:
        df[col] = df[col] / df['Minutes / Div By 90_y']

###filling Nan
df.loc[(df["Matches Played_y"] == 0), 'Matches Played_y'] = df['Matches Played_y'].mean()
df.loc[(df["Minutes / Div By 90_y"] == 0), 'Minutes / Div By 90_y'] = df['Minutes / Div By 90_y'].mean()

for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    df[i].fillna(df[i].mean(),inplace=True)

###Normalize####################

df = df.drop(['Freq Inj'], axis=1)

normToAge = ["Career INJ days","Career Suspension days"]
normToFiveMonth = ["inj22","Sus days 22"]
normToYaer = ["inj days 19-20","inj days 20-21","inj days 18-19","sus days 20-21","sus days 19-20","sus days 18-19",]
normToMinMax = ["Suspended","Knock","Hamstring","Ankle/Foot Injury","Knee Injury","Thigh Muscle Strain","Groin/Pelvis Injury","Illness","Virus",
                "Calf/Shin Injury","Groin Strain","Back Injury","Hip/Thigh Injury","Calf Muscle Strain","Thigh Muscle Rupture",
                "Shoulder Injury","Head Injury","Foot Injury","ACL Knee Ligament Injury","Ankle Ligaments","MCL Knee Ligament Injury",
                "Concussion","Sprained Ankle","Rib Injury","Toe Injury","Achilles tendonitis","Thumb / Wrist Injury"]
print(df.head().to_string())
for col in normToYaer:
    df[col] = df[col]/365
for col in normToFiveMonth:
    df[col] = df[col]/150
for col in normToAge:
    df[col] = df[col] / ((df['age']-16)*365)
scaler = MinMaxScaler()
for col in normToMinMax:
    df[normToMinMax] = scaler.fit_transform(df[normToMinMax])
df.to_csv('26-1Norm3.csv', index=False)




# df2.to_csv('AllStas16.csv', index=False)

#df['Full_name'] =(df['first'].fillna('') + ' ' + df['last'].fillna('')+' ' +df['third'].fillna('')).str.strip(' ')
#df['Full_name'] =(df['Name'].apply(lambda x: x.split()[0])+' '+df['Name'].fillna('')).str.strip(' ').apply(lambda x: x.split()[-1])
#final['Prefix'] = final['Story'].apply(lambda x: x.split()[-1])
#df_inj = pd.read_csv('Allinj11-1.csv')
#df_inj['First Name'] = df_inj['First Name'] .str.replace('-',' ')
#df_inj['Last Name'] = df_inj['Last Name'] .str.replace('-',' ')

#df_inj['Full_name'] =(df_inj['First Name'].apply(lambda x: x.split()[0])+' '+df_inj['Last Name'].fillna('')).apply(lambda x: x.split()[-1])
#df_names_test['First']=df_names_test['Name'].apply(lambda x: x.split(" ")[0])
#df_inj['Name'] = df_inj[['First Name', 'Last Name']].apply(lambda x: ' '.join(x), axis=1)
#df_inj = df_inj.sort_values(by=['Name'])


######Normalize All
#
# df = pd.read_csv(r'25-1Norm2.csv',sep=',')
# df = df.drop(['Freq Inj'], axis=1)
#
# normToTimeX = ["Goals_x","Total Shots_x","Passes Total Distance_x","Asissts_x","Yellows_x","Reds_x","Passes Attemted_x","Short Passes_x","Medium Passes_x",
#               "Long Passes_x","Cross Passes_x","Head Pases_x","Tackeled_x","Tackeled Dribbels_x","Pressure Applying_x","Blocks_x",
#               "Shots Blocks_x","Clearence_x","Touches_x","Touches Defencive Penalty Area_x","Touches Defencive 3RD_x","Touches Defencive 3RD_x",
#               "Touches Mid 3RD_x","Touches Attacking 3RD_x","Touches Attacking Penalty Area_x","Touches Live Game_x","Dribbels Succes_x",
#               "Dribbels Succes_x","Dribbels Attempted_x","Carries_x","Carries To Distance_x","Carries Proggresive_x",
#               "Carries Proggresive Foword_x","Carries Tackeled_x","Reciving Passes_x","Fouls Commited_x","Fouls Drawn_x",
#               "Tackele Won_x","looseBallRecovered_x","Aerial Duals Won_x","Aerial Duals Lost_x"]
#
# normToTimeY = ["Goals_y","Total Shots_y","Passes Total Distance_y","Asissts_y","Yellows_y","Reds_y","Passes Attemted_y","Short Passes_y","Medium Passes_y",
#               "Long Passes_y","Cross Passes_y","Head Pases_y","Tackeled_y","Tackeled Dribbels_y","Pressure Applying_y","Blocks_y",
#               "Shots Blocks_y","Clearence_y","Touches_y","Touches Defencive Penalty Area_y","Touches Defencive 3RD_y","Touches Defencive 3RD_y",
#               "Touches Mid 3RD_y","Touches Attacking 3RD_y","Touches Attacking Penalty Area_y","Touches Live Game_y","Dribbels Succes_y",
#               "Dribbels Succes_y","Dribbels Attempted_y","Carries_y","Carries To Distance_y","Carries Proggresive_y",
#               "Carries Proggresive Foword_y","Carries Tackeled_y","Reciving Passes_y","Fouls Commited_y","Fouls Drawn_y",
#               "Tackele Won_y","looseBallRecovered_y","Aerial Duals Won_y","Aerial Duals Lost_y"]
# normToAge = ["Career INJ days","Career Suspension days"]
# normToFiveMonth = ["inj22","Sus days 22"]
# normToYaer = ["Christmas-played-games","Number of Inj","inj days 19-20","inj days 20-21","inj days 18-19","sus days 20-21","sus days 19-20","sus days 18-19",]
# normToMinMax = ["age","Matches Played_x","Minutes / Div By 90_x","Matches Played_y","Minutes / Div By 90_y","Suspended","Knock","Hamstring","Ankle/Foot Injury","Knee Injury","Thigh Muscle Strain","Groin/Pelvis Injury","Illness","Virus",
#                 "Calf/Shin Injury","Groin Strain","Back Injury","Hip/Thigh Injury","Calf Muscle Strain","Thigh Muscle Rupture",
#                 "Shoulder Injury","Head Injury","Foot Injury","ACL Knee Ligament Injury","Ankle Ligaments","MCL Knee Ligament Injury",
#                 "Concussion","Sprained Ankle","Rib Injury","Toe Injury","Achilles tendonitis","Thumb / Wrist Injury"]
#
# for col in normToYaer:
#     df[col] = df[col]/365
# for col in normToFiveMonth:
#     df[col] = df[col]/150
# for col in normToAge:
#     df[col] = df[col] / ((df['age']-16)*365)
# scaler = MinMaxScaler()
# for col in normToMinMax:
#     df[normToMinMax] = scaler.fit_transform(df[normToMinMax])
# for col in normToTimeY:
#     df[normToTimeY] = scaler.fit_transform(df[normToTimeY])
# for col in normToTimeX:
#     df[normToTimeX] = scaler.fit_transform(df[normToTimeX])
# df.to_csv('26-2AllNorm.csv', index=False)


####Merging Dummies#####
df.loc[(df["Position_FW"] == 1), 'Position'] = 'FW'
df.loc[(df["Position_AMF"] == 1), 'Position'] = 'AMF'
df.loc[(df["Position_DF"] == 1), 'Position'] = 'DF'
df.loc[(df["Position_DMF"] == 1), 'Position'] = 'DMF'
df.loc[(df["Position_MF"] == 1), 'Position'] = 'MF'