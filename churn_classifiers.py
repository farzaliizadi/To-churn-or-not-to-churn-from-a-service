# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:15:47 2019

@author: Izadi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir(r'D:\desktop\Python_DM_ML_BA')
df = pd.read_csv('churn.csv')
df.shape
df.head()
df.columns
df.info()
d = df.describe()
df.isnull().sum()
df.isnull().sum().sort_values(ascending=False)
ch = df['Churn'].value_counts()
ch
# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.  
sns.countplot(df['Churn'], hue=df['Churn']) 
plt.xlabel('Churn', size=20)
plt.ylabel('Counts', size=20)
plt.title('CHURN Count', size=20)
plt.show()  

# Group df by 'Churn' and compute the mean
m = df.groupby(['Churn']).mean()

m.plot(kind='bar')
plt.xlabel('Churn', size=50)
plt.ylabel('Counts', size=50)
plt.title('CHURN MEAN', size=50)
plt.legend(loc=1)

# Count the number of churners and non-churners by State
state = df.groupby('State')['Churn'].value_counts()
state 
# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.  
sns.countplot(df['State'], hue=df['Churn']) 
plt.xlabel('Churn', size=20)
plt.ylabel('Counts', size=20)
plt.title('CURN VALUE COUNTS', size=20)
plt.show()  
# 1. Befor anything we have to do explatory data analysis. 
# To do so we have to transform object varibles to numeric and drop unnecessary features.
df = df.drop(df[['Area_Code','Phone']], axis=1)
df['Churn'] = df.Churn.map({'no': 0 , 'yes': 1})
df['Vmail_Plan'] = df['Vmail_Plan'].map({'no': 0 , 'yes': 1})
df['Intl_Plan'] = df['Intl_Plan'].map({'no': 0 , 'yes': 1})
df['Vmail_Plan'].head()
df['Churn'].head()
df.head()
df.info()
# 2. Now the only object column is the State column
#Next we separate numerical from carogorical features
numeric_cols = [x for x in df.dtypes.index if df.dtypes[x]!='object']
cat_cols = [x for x in df.dtypes.index if df.dtypes[x]=='object']
# 3.Then we transform the state column by LabelEncoder to numeric
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()

for col in cat_cols:
    df[col] = labelEncoder.fit_transform(df[col])

mportant thing is to get the correlation among features.
#Also corr 
c = df.corr()
c
# 7. Use the heat map to see the big correlations'''
sns.set(style='white')
plt.figure(figsize=(12, 8))
#
plt.title('CORRELATION')                        #best
# Create the heatmap
sns.heatmap(c, annot=True, cmap='BuGn',fmt='.0%')
plt.xticks(rotation=20)
plt.show()
''' We see that correlations among durations and charges
(Intl_Mins, Intl_Charge)=100%
(Night_Mins, Night_Charge)=100%
(Eve_Mins, Eve_Charge) =100%
(Day_Mins, Day_charge)=100%
(Vmail_Massage, Vmail_Plan)=96% 
SO we should drop one of the above pairs 
as no extra information arise from keeping both.
'''
dg = df.drop(['Intl_Charge', 'Night_Charge','Eve_Charge','Day_Charge','Vmail_Plan'], axis=1)
b = dg.corr()
dg.columns
dg.shape
dg.head()
''' On the other hand the correlations between target feature Churn and 
the remaing feature are quite low which tell us their are weak predictors.
0.017, -0.09, 0.21, 0.09, 0.04,
0.07, 0.21, 0.26, 0.02,
0.01, 0.01, -0.1, 0.01

'''
sns.set(style='white')
plt.figure(figsize=(12, 8))
#
plt.title('CORRELATION')                        #best
# Create the heatmap
sns.heatmap(b, annot=True, cmap='BuGn',fmt='.0%')
plt.xticks(rotation=20)
plt.show()
# 8. Checking and removing outliers if any by using boxplot.
# Create the box plot
dg.columns
dg.boxplot(column=['Account_Length','Day_Mins','Eve_Mins','Night_Mins'] )
dg.boxplot(column=['Day_Calls','Eve_Calls', 'Night_Calls'] , patch_artist=True)
dg.boxplot(column=['Intl_Mins','CustServ_Calls'], patch_artist=True)
dg.boxplot(column=['Churn','Intl_Plan'], patch_artist=True)
dg.boxplot(column=['Vmail_Message'], patch_artist= True)
dg.boxplot(column=['Intl_Calls'], patch_artist= True)
# By droping the following outlies, we get 
dg.drop(dg[(dg.Intl_Mins > 17.4) | (dg.Intl_Mins < 3.2)].index, inplace=True)
dg.drop(dg[dg.Vmail_Message > 50].index, inplace=True)
dg.drop(dg[dg.Intl_Calls > 10].index, inplace=True)
dg.drop(dg[dg.CustServ_Calls > 3.7].index, inplace=True)
dg.drop(dg[dg.Account_Length > 208].index, inplace=True)
dg.drop(dg[(dg.Day_Mins > 323) | (dg.Day_Mins < 48)].index, inplace=True)
dg.drop(dg[(dg.Eve_Mins > 338) | (dg.Eve_Mins < 63)].index, inplace=True)   
dg.drop(dg[(dg.Night_Mins > 338) | (dg.Night_Mins < 66)].index, inplace=True)
dg.drop(dg[(dg.Day_Calls > 154) | (dg.Day_Calls < 51)].index, inplace=True)
dg.drop(dg[(dg.Eve_Calls > 152) | (dg.Eve_Calls < 47)].index, inplace=True)   
dg.drop(dg[(dg.Night_Calls > 152) | (dg.Night_Calls < 47)].index, inplace=True)
dg.to_csv('Churn2.csv', index=False) 
dg = pd.read_csv('Churn2.csv')      
dg.head()
dg.shape
dg.columns

# We can compare both DataFrames to see what are change regarding the outliers. 
sns.boxplot(data = df)
plt.show()

sns.boxplot(data = dg)
plt.show()


# We see that there is no any outliers.
# Create the box plot
sns.boxplot(x = 'Churn',
            y = 'CustServ_Calls',
            data = dg)
plt.show()

# Add "Intl_Plan" as a third variable
sns.boxplot(x = 'Churn',
            y = 'CustServ_Calls',
            data = dg,
            hue = "Intl_Plan")
plt.show()

# 9. Now it is time to check normality distribution of the columns.
sns.pairplot(dg, hue="Churn", palette="husl", markers=["o", "s"])
sns.pairplot(dg, hue="Churn", markers=["o", "s"])
sns.pairplot(dg, diag_kind="kde")
sns.pairplot(dg, kind="reg")
#################################################################
# Since the values of Churn is yes or no, it is a classification problem.

X = dg.drop(['Churn'], axis=1)
y = dg['Churn']
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(X.columns, importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(X.columns)
plt.xlim([-1, X.shape[1]])
plt.show()

###########################################################
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=250, random_state=0)
gbc.fit(X, y)
importances = gbc.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


for f in X.columns:
    print(f)
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(X.columns, importances[indices], color="g")
plt.xticks(X.columns)
plt.xlim([-1, X.shape[1]])
plt.show()
# Model building with 8 different classifiers which the score, confution_matrix,
#  classification_report as well as ROC curve are all perfect.
np.random.seed(37)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled =  sc.fit_transform(X)
#split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.26,random_state=0)
from sklearn.metrics import accuracy_score,roc_curve ,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
from sklearn.ensemble import ExtraTreesClassifier
exc = ExtraTreesClassifier()
from sklearn.naive_bayes import GaussianNB
model_naive = GaussianNB()
from sklearn.svm import SVC
svm_model= SVC(gamma='scale')
from sklearn.neighbors import KNeighborsClassifier as KNN
kn = KNN()
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
from xgboost import XGBClassifier
xgb = XGBClassifier()
from sklearn.ensemble import GradientBoostingClassifier
gbr = GradientBoostingClassifier()
from sklearn.ensemble import BaggingClassifier 
bc = BaggingClassifier() 

models = [lg, dtr, rfc, exc, model_naive, svm_model, kn, ada, xgb, gbr,bc]


modnames = ['LogisticRegression', 'DecisionTreeClassifier','RandomForestClassifier',
            'ExtraTreesClassifier', 'GaussianNB', 'SVC', 'KNeighborsClassifier',
            'AdaBoostClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 'BaggingClassifier']

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix(y_test,y_pred)
    print('The accuracy of ' + modnames[i] + ' is ' + str(accuracy_score(y_test,y_pred)))
    print('The confution_matrix ' + modnames[i] + ' is ') 
    print(str(confusion_matrix(y_test,y_pred)))
    
''' 
The accuracy of LogisticRegression is 0.9005524861878453
The confution_matrix LogisticRegression is 
[[636  12]
 [ 60  16]]
The accuracy of DecisionTreeClassifier is 0.919889502762431
The confution_matrix DecisionTreeClassifier is 
[[614  34]
 [ 24  52]]
The accuracy of RandomForestClassifier is 0.9516574585635359
The confution_matrix RandomForestClassifier is 
[[647   1]
 [ 34  42]]
The accuracy of ExtraTreesClassifier is 0.9406077348066298
The confution_matrix ExtraTreesClassifier is 
[[648   0]
 [ 43  33]]
The accuracy of GaussianNB is 0.8798342541436464
The confution_matrix GaussianNB is 
[[607  41]
 [ 46  30]]
The accuracy of SVC is 0.9447513812154696
The confution_matrix SVC is 
[[646   2]
 [ 38  38]]
The accuracy of KNeighborsClassifier is 0.9116022099447514
The confution_matrix KNeighborsClassifier is 
[[641   7]
 [ 57  19]]
The accuracy of AdaBoostClassifier is 0.893646408839779
The confution_matrix AdaBoostClassifier is 
[[618  30]
 [ 47  29]]
The accuracy of XGBClassifier is 0.962707182320442
The confution_matrix XGBClassifier is 
[[644   4]
 [ 23  53]]
The accuracy of GradientBoostingClassifier is 0.9516574585635359
The confution_matrix GradientBoostingClassifier is 
[[639   9]
 [ 26  50]]
The accuracy of BaggingClassifier is 0.9571823204419889
The confution_matrix BaggingClassifier is 
[[640   8]
 [ 23  53]]
 
We see that both 
RandomForestClassifier and  GradientBoostingClassifier are champion both in accuracy and confution matrix.
'''   
for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = classification_report(y_test,y_pred)
    print('auc_roc_R ' + modnames[i] + ' is ' )
    print(str(auc_roc))
# only ROC curves for  last model 
y_pred = gbr.predict(X_test)    
from sklearn.metrics import roc_curve, auc,confusion_matrix,roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')   
 #############################################   




