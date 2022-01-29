
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt      
import math
import seaborn as sns
import seaborn as sb

from collections import Counter
from sklearn.decomposition import PCA
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv(r'../LogisticRegModel/25-1Norm2.csv', sep=',')
my_df = df[df["CrsINJURY"] == 1]
obj = df.groupby('Team')['Christmas-played-games'].mean()
obj2 = df.groupby('Team')['CrsINJURY'].sum()
obj3 = df.groupby('Team')['Christmas-played-games'].sum()

#print(obj)
obj1 = df.groupby('Team')['Name'].count()
#print(obj1)


# In[ ]:





# In[114]:


df_s = pd.read_csv(r'For_scattaring.csv',sep=',')

sns.relplot(x='Amount-of-players',y='Crs_inj',hue='Christmas-played-games',sizes=(40,400),data=df_s)


# In[9]:


plt.ylabel('Number-Of-Teams')
plt.xlabel('CrsINJURY')

obj2.value_counts().plot(kind='bar',color='blue')


# In[24]:


obj2.value_counts().plot(kind='pie')


# In[11]:


plt.ylabel('Number-Of-Teams')
plt.xlabel('Christmas-played-games')

obj.value_counts().plot(kind='bar',color='green')


# In[25]:


obj.value_counts().plot(kind='pie')


# In[13]:


ct=pd.crosstab(obj,obj2,normalize='index')
ct


# In[22]:


ct.plot(kind='bar')
plt.ylabel('Precent-of-Teams')


# In[33]:





# In[7]:


def cross_tabulation(df, col_name, other_col_name):
    ct=pd.crosstab(df[col_name],df[other_col_name],normalize='index')
    return ct
def plot_cross_tabulation(df, col_names, other_col_name):
    fig,axes=plt.subplots(1,len(col_names),figsize=(20,5))
    i=0
    for ax in axes:
        ct=cross_tabulation(df[:30],col_names[i],other_col_name)
        ct.plot.line(rot=0,ax=ax)
        i+=1


# In[8]:


plot_cross_tabulation(df,['Christmas-played-games','Tackeled_x'],'CrsINJURY')#רואים שאין קורולציה


# In[9]:


plot_cross_tabulation(df,['Matches Played_x','Number of Inj'],'CrsINJURY')#ישנה קורולציה 


# In[10]:


def get_highly_correlated_cols(df):
    df_corr=df.corr(method='pearson')
    correlations=[]
    tuple_arr=[]
    cols=[]
    for col_name in df_corr.columns:
        cols.append(col_name)
    for i in range(0,len(cols)):
        for j in range(i+1,len(cols)):
            if df_corr[cols[i]][cols[j]]>=0.0:
             tuple_arr.append((i,j))
             correlations.append(df_corr[cols[i]][cols[j]])
    return correlations, tuple_arr


# In[12]:


cols_for_correlations=['Matches Played_x','inj days 20-21','inj22','Freq Inj','Christmas-played-games','Number of Inj']
correlations, tuple_arr = get_highly_correlated_cols(df[cols_for_correlations])
#print(correlations)

indx_sort = np.argsort(correlations)


for n_correlation in indx_sort:
    col_lt, col_rt = tuple_arr[n_correlation]
    col_name_lt, col_name_rt = cols_for_correlations[col_lt], cols_for_correlations[col_rt]
    title = "corr('%s', '%s')=%4.2f" %(col_name_lt, col_name_rt, correlations[n_correlation]) 
    #print(title)
    


# In[13]:


def plot_high_correlated_scatters(df):
    correlation,tuple_arr=get_highly_correlated_cols(df)
    fig, axes= plt.subplots(1,len(tuple_arr),figsize=(20,5))
    i=0
    cols=[]
    for col_name in df.columns:
        cols.append(col_name)
    for ax in axes:
        ax.scatter(df[cols[tuple_arr[i][0]]],df[cols[tuple_arr[i][1]]])
        ax.set_xlabel("corr('%s','%s)=%4.2f" %(cols[tuple_arr[i][0]],cols[tuple_arr[i][1]],correlation[i]),labelpad=5)
        i+=1


# In[14]:


cols_for_correlations = ['age','CrsINJURY','inj22']
plot_high_correlated_scatters(df[cols_for_correlations])


# In[15]:


cols_for_correlations = ['Matches Played_x','CrsINJURY'
                         ,'Christmas-played-games']
plot_high_correlated_scatters(df[cols_for_correlations])


# In[16]:


cols_for_correlations = ['inj days 20-21','Hamstring','Ankle/Foot Injury']
plot_high_correlated_scatters(df[cols_for_correlations])


# In[17]:


cols_for_correlations = ['inj days 20-21','Foot Injury','ACL Knee Ligament Injury']
plot_high_correlated_scatters(df[cols_for_correlations])


# In[ ]:




