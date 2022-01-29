import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

########first use of the model####################

df = pd.read_csv(r'../DataCleaning/fullDF/16-1FirstFULLclnDF.csv', sep=',')
df_f = df.columns[df.columns != 'CrsINJURY']

X = df[df_f]
X= X.drop(columns=['Name', 'Team'])
y =df[['CrsINJURY']].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5)

m_model = LogisticRegression(solver='liblinear').fit(X_train, y_train)

y_pred = m_model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


print(m_model.score(X_test, y_test))
my_df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
print(my_df.to_string())




#################get best values:################################################

df = pd.read_csv(r'25-1Norm2.csv', sep=',')

X = df.loc[:, df.columns != 'CrsINJURY']
y = df.loc[:, df.columns == 'CrsINJURY'].values.ravel()

X = X.drop(columns=['Name', 'Team'])

os1 = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os1_data_X,os1_data_y= os1.fit_resample(X_train, y_train)
os1_data_X = pd.DataFrame(data=os1_data_X,columns=columns )
os1_data_y= pd.DataFrame(data=os1_data_y,columns=['CrsINJURY'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os1_data_X))
print("Number of no subscription in oversampled data",len(os1_data_y[os1_data_y['CrsINJURY']==0]))
print("Number of subscription",len(os1_data_y[os1_data_y['CrsINJURY']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os1_data_y[os1_data_y['CrsINJURY']==0])/len(os1_data_X))
print("Proportion of subscription data in oversampled data is ",len(os1_data_y[os1_data_y['CrsINJURY']==1])/len(os1_data_X))




X = df.loc[:, df.columns != 'CrsINJURY']
X = X.drop(columns=['Name', 'Team'])
y = df.loc[:, df.columns == 'CrsINJURY'].values.ravel()


columns = X_train.columns
os1_data_X,os1_data_y= os1.fit_resample(X_train, y_train)
os1_data_X = pd.DataFrame(data=os1_data_X,columns=columns )

logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(os1_data_X, os1_data_y)
print(rfe.support_)
print(rfe.ranking_)

X = X.iloc[:, [1,2,5,9,14,15,16,19,20,21,22,23,24,27,28,29,31,32,35,39,40,41,43,45,51,53,54,55,57,58,60,61,62,63,64,66,67,69,72,
            80,81,82,84,87,88,89,90,91,93,100,105,106,107,108,109,110,111,113,114,116,117,118,123,124,125]]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
m_model = LogisticRegression(solver='liblinear').fit(X_train, y_train)
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
y_pred = m_model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(m_model.score(X_test, y_test)))
cf_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))

###########plot CNF#################
group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='OrRd_r')

ax.set_title('Model Confusion matrix')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()


#########model plot second Version########

def plot_cm(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = 'True\n%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = 'False\n%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="OrRd_r", annot=annot, fmt='', ax=ax)


plot_cm(y_test, y_pred)
plt.show()
####test2#############
#df = pd.read_csv(r'26-2AllNorm.csv',sep=',')
#X = X.iloc[:, [1,2,3,5,9,14,16,17,18,21,22,24,27,29,30,31,33,34,35,36,38,39,40,42,43,45,48,51,53,57,61,62,63,64,67,69,72,73,74,75,78,82,85,87,88,89,
#               92,101,104,105,106,107,108,110,112,113,115,116,117,118,122,123,126,127]]


####tets3#################
#df = pd.read_csv(r'26-1NormTry.csv',sep=',')
#X = X.iloc[:, [2,4,5,7,9,10,12,13,15,16,17,19,21,22,23,24,25,27,28,30,31,36,39,40,42,44,45,48,49,51,54,56,57,58,61,62,63,64,65,66,67,68,69,72,76,
#               77,82,83,84,85,89,90,104,105,106,107,108,111,112,113,114,116,117,118]]
