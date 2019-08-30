# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 02:19:31 2019

@author: Jesus kid
"""

import os
os.chdir(r'C:\Users\Jesus kid\Desktop\python project')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv('Store_Train_Forecast_Class.csv')




'''Data cleaning 1 - fill nulls (part 2 will be after the plot section)'''

train.describe()
train.isnull().sum()

#Check the column list
for i in range(0,len(train.columns.tolist())):
    print(train.columns[i])

# fill missing data
train.Item_Weight.fillna(train.Item_Weight.mean(), inplace=True)
train.Outlet_Size.fillna('Medium',inplace=True)

#clean up the var'Item_Fat_Content'
train=train.replace('LF', 'Low Fat')
train=train.replace('Low fat', 'Low Fat')
train=train.replace('low fat', 'Low Fat')
train=train.replace('reg', 'Regular')



'''plot section 1 (bar plot of categorical variable (unique values) vs counts, and vs sales median)'''

'''outlet_Location_Type'''
x1=train.Outlet_Location_Type
x1.value_counts().plot(kind='bar')
plt.xlabel('Outlet_Location_Type', fontsize=16)
plt.ylabel('count',fontsize=16)
plt.title('Outlet Location Type count',fontsize=18)
plt.show()

train.groupby(['Outlet_Location_Type'])["Item_Outlet_Sales"].median().plot(kind='bar')
plt.xlabel('Outlet_Location_Type', fontsize=16)
plt.ylabel('sales',fontsize=16)
plt.title('Outlet Location Type sales median',fontsize=18)
plt.show()



'''Outlet_Size'''
x2=train.Outlet_Size
x2.value_counts().plot(kind='bar')
plt.xlabel('Outlet_Size',fontsize=16)
plt.ylabel('count',fontsize=16)
plt.title('Outlet Size count',fontsize=18)
plt.show()

train.groupby(['Outlet_Size'])["Item_Outlet_Sales"].median().plot(kind='bar')
plt.xlabel('Outlet_Size',fontsize=16)
plt.ylabel('sales',fontsize=16)
plt.title('Outlet_size sales median', fontsize=18)
plt.show()



'''Outlet_Type'''
x3=train.Outlet_Type
x3.value_counts().plot(kind='bar')
plt.xlabel('Outlet_Type',fontsize=16)
plt.ylabel('count',fontsize=16)
plt.title('Outlet Type count',fontsize=18)
plt.show()

train.groupby(['Outlet_Type'])["Item_Outlet_Sales"].median().plot(kind='bar')
plt.xlabel('Outlet_Type',fontsize=16)
plt.ylabel('sales',fontsize=16)
plt.title('Outlet_Type sales median', fontsize=18)
plt.show()



'''Item_Fat_Content'''
x4=train.Item_Fat_Content
x4.value_counts().plot(kind='bar')
plt.xlabel('Item_Fat_Content',fontsize=16)
plt.ylabel('count',fontsize=16)
plt.title('Item Fat Content count', fontsize=18)
plt.show()

train.groupby(['Item_Fat_Content' ])["Item_Outlet_Sales"].median().plot(kind='bar')
plt.xlabel('Item_Fat_Content',fontsize=16)
plt.ylabel('sales',fontsize=16)
plt.title('Item Fat Content sale median', fontsize=18)
plt.show()



'''Item_Type'''
x6a=train.Item_Type.value_counts().nsmallest(5)
x6b=train.Item_Type.value_counts().nlargest(5)
'''Item Type count(low 5)'''
x6a.plot(kind='bar')
plt.xlabel('Item_Type',fontsize=13)
plt.ylabel('count',fontsize=16)
plt.title('Item Type count(low 5)',fontsize=15)
plt.show()

train.groupby(['Item_Type' ])["Item_Outlet_Sales"].median().nsmallest(5).plot(kind='bar')
plt.xlabel('Item_Type',fontsize=13)
plt.ylabel('sales',fontsize=16)
plt.title('Item Type sales median(low 5)',fontsize=15)
plt.show()


'''Item Type count(high 5)'''
x6b.plot(kind='bar')
plt.xlabel('Item_Type',fontsize=13)
plt.ylabel('count',fontsize=16)
plt.title('Item Type count(high 5)',fontsize=15)
plt.show()

train.groupby(['Item_Type' ])["Item_Outlet_Sales"].median().nlargest(5).plot(kind='bar')
plt.xlabel('Item_Type',fontsize=13)
plt.ylabel('sales',fontsize=16)
plt.title('Item Type sales median(high 5)',fontsize=15)
plt.show()








'''plot section 2 (scatterplot of numerical variable vs sales median)'''
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight vs Item_Outlet_Sales")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"],'.', alpha = 0.3)


plt.figure(figsize=(12,7))
plt.xlabel('Item_Visibility')
plt.ylabel('Item_Outlet_Sales')
plt.title('Item Visibility vs Item Outlet Sales ')
plt.plot(train.Item_Visibility, train['Item_Outlet_Sales'],'.', alpha = 0.3)


plt.figure(figsize=(12,7))
plt.xlabel('Item_MRP')
plt.ylabel('Item_Outlet_Sales')
plt.title('Item MRP vs Item Outlet Sales')
plt.plot(train.Item_MRP, train['Item_Outlet_Sales'],'.', alpha = 0.3)







'''data cleaning 2- create dummy variable, drop useless variable'''

train_2=train.copy(deep=True)
train_2.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'], axis='columns', inplace=True)

'''correlation HeatMap among the numerical variables'''
import seaborn as sns
import math

plt.figure(figsize=(15,15))
cor = abs(train_2.corr())
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#create dummy variables for categorical variables
train_2= pd.get_dummies(train_2,columns=['Item_Fat_Content', 'Outlet_Location_Type','Outlet_Size',
                                     'Outlet_Type','Item_Type'], drop_first=True )


#Splitting X and Y
y=train_2.Item_Outlet_Sales 

X=train_2.drop(['Item_Outlet_Sales'], axis=1)
X_dummy=X.drop(['Item_Weight','Item_Visibility','Item_MRP'], axis=1)



#Data Preprocessing (its a part of the machine lerning)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_scaled = pd.DataFrame(sc_X.fit_transform(X[['Item_Weight','Item_Visibility','Item_MRP']]), columns=X[['Item_Weight','Item_Visibility','Item_MRP']].columns)
X_scaled= pd.concat([X_scaled, X_dummy], axis=1, join='inner').values
X_scaled=pd.DataFrame(X_scaled, columns=X.columns)

sc_y = StandardScaler()
y_scaled = sc_y.fit_transform(pd.DataFrame(y))
y_scaled=pd.DataFrame(y_scaled, columns=["Item_Outlet_Sales"])





'''machine learning modeling'''

'''feature engineering (find the variables that gives max R2 accuracy score)'''
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

estimator = LinearRegression() #use regression model for regression problem
list_r2=[]
max_r2 = 0
for i in range(1,len(X_scaled.loc[0])+1):
    selector = RFE(estimator, i, step=1)
    selector = selector.fit(X_scaled, y_scaled)
    adj_r2 = 1 - ((len(X_scaled)-1)/(len(X_scaled)-i-1))*(1-selector.score(X_scaled, y_scaled))
    list_r2.append(adj_r2)# mse = 
    if max_r2 < adj_r2:
        sel_features = selector.support_
        max_r2 = adj_r2
       
X_sub = X_scaled.iloc[:,sel_features]
X_sub.columns.tolist() #selected features

#split training  and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_sub,y,random_state=0)






​
'''Machine Learning algorithms for regression---------------------------------------------------------------------'''
'''fit linear regression to the model'''
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(X_train, y_train)
y_pred = lr1.predict(X_test)
print(lr1.coef_, lr1.intercept_) 

'''R2 and adjusted R2 , and rmse'''
from sklearn.metrics import mean_squared_error
import math
r2=lr1.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-lr1.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred) #biased mean
rmse = math.sqrt(mse)
print(rmse)

'''summary table for coefficients'''
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as lr
X2_train=sm.add_constant(X_train) # add a column of 1 beside x col
ols=sm.OLS(y_train.astype(float),X2_train.astype(float))# ordinary least square = linear regression
lr=ols.fit() 
print(lr.summary())






'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''fit knn to the model'''
from sklearn.neighbors import KNeighborsRegressor #KNeighborsRegressor if linear regression
knn = KNeighborsRegressor() 

'''find the optimal parameters in KNN'''
param_dict = {
                'n_neighbors': [5,10,15],
                'weights': ['uniform', 'distance' ],
                'p' :[1, 2]          
             }

from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn,param_dict)
knn.fit(X_train,y_train)
knn.best_params_ 
knn.best_score_


'''refit knn to the model with optimal parameters'''
knn=KNeighborsRegressor(n_neighbors= 15, p=1, weights='uniform')
knn.fit(X_train,y_train)
#predictions for test
y_pred2 = knn.predict(X_test)


'''R2 and adjusted R2, and rmse'''
r2=knn.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-knn.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred2) #biased mean
rmse = math.sqrt(mse)
print(rmse)







'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''fit svr to the model'''
from sklearn.svm import SVR # svc= svm for classification; SVR= svm for regressor
svr=SVR()



'''find the optimal parameters in KNN (this gonna take a long time!!!!!!!)'''
param_dict = {
                'kernel': ['linear', 'poly', 'rbf'],
                'degree': [2,3,4],
                'C' :[0.001, 0.01, 0.1, 1]          
             }


from sklearn.model_selection import GridSearchCV
svr = GridSearchCV(svr,param_dict)
svr.fit(X_train,y_train)
svr.best_params_ # best parameters for SGDClassifier(), you put the output parameters in SGDClassifier()
svr.best_score_ # the above parameters gives you this best accuracy



'''refit svr to the model with optimal parameters'''
svr=SVR(C=1, degree=2, kernel='linear')
svr.fit(X_train,y_train)
#predictions for test
y_pred3 = svr.predict(X_test)
 

'''R2 and adjusted R2, and rmse'''
r2=svr.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-svr.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred3) #biased mean
rmse = math.sqrt(mse)
print(rmse)








'''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
'''random forest regressor'''

''' plot, to find the optimal n_estimator(68) and  max_depth(5)'''
from sklearn.ensemble import RandomForestRegressor

test_scores=[]
for n in range(15,70):
    model=RandomForestRegressor(n_estimators=n, max_depth=10, random_state=0)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores.append(mean_squared_error(y_test, y_pred))
    
plt.plot(range(15, 70), test_scores)
plt.xlabel('n of DTs')
plt.ylabel('MSE')



test_scores1=[]
for k in range(1,60):
    model=RandomForestRegressor(n_estimators=68, max_depth=k, random_state=0)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    test_scores1.append(mean_squared_error(y_test, y_pred))
    
plt.plot(range(1, 60), test_scores1)
plt.xlabel('n of depth')
plt.ylabel('MSE')


'''fit rfr to the model'''
rfr= RandomForestRegressor(n_estimators=68, criterion='mse', max_depth=5)
#20 decision tree
rfr.fit(X_train,y_train)
y_pred4 = rfr.predict(X_test)


'''R2 and adjusted R2, and rmse'''
r2=rfr.score(X_test,y_test)
print(r2)
adj_r2 = 1 - ((len(X_test)-1)/(len(X_test)-i-1))*(1-rfr.score(X_test, y_test))
print(adj_r2)

mse=mean_squared_error(y_test, y_pred4) #biased mean
rmse = math.sqrt(mse)
print(rmse)

