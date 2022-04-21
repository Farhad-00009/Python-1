#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import plotly.express as px





# In[10]:


data = pd.read_csv('datasets_137197_325031_test_Y3wMUE5_7gLdaTN.csv')
data.head(10)


# In[8]:


data.shape


# In[5]:


data.columns=data.columns.str.lower()


# In[6]:


data.columns=["loan_id", "gender", "married", "dependents", "education","self_employed", "applicant_income", "co-applicant_income", "loan_amount","credit_history","property_area","loan_status"] 


# In[7]:


data.isnull().sum()


# In[8]:


data['loan_amount']=data['loan_amount'].fillna(data['loan_amount'].mean())  


# In[9]:


data['credit_history']=data['credit_history'].fillna(data['credit_history'].median())


# In[10]:


data.dropna(axis=0, inplace=True)


# In[11]:


data.isnull().sum()


# In[12]:


data.head()


# In[13]:


data.shape


# In[14]:


data.info()


# In[15]:


data.describe()


# In[16]:


type(data['dependents'].iloc[0])


# In[17]:


data['dependents'].unique()


# In[18]:


model6=LabelEncoder()


# In[19]:


model6.fit(data['dependents'])


# In[20]:


data['dependents']= model6.transform(data['dependents'])


# In[21]:


data[data['loan_status']=='Y'].count()['loan_status']


# In[22]:


data[data['loan_status']=='N'].count()['loan_status']


# In[23]:


plt.figure(figsize=(8,8))
plt.pie(x=[376,166], labels=['Yes','No'], autopct='%1.0f%%', pctdistance=0.5,labeldistance=0.7,colors=['y','m'])
plt.title('Distribution of Loan Status')


# In[24]:


##69% of applicants repay the loan and 39% do not repay the loan.


# In[25]:


plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.countplot(x="gender" ,hue='loan_status',palette='plasma')

plt.subplot(2,3,2)
sns.countplot(x='married',hue='loan_status',palette='viridis')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,3)
sns.countplot(x='education',hue='loan_status',palette='copper')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,4)
sns.countplot(x='credit_history',hue='loan_status',palette='summer')

plt.subplot(2,3,5)
sns.countplot(x='self_employed',hue='loan_status',palette='autumn')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,6)
sns.countplot(x='property_area',hue='loan_status',palette='PuBuGn')
plt.ylabel(' ')
plt.yticks([ ])


# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.violinplot(x='gender', y='loan_amount',hue='loan_status', d=data,palette='plasma')

plt.subplot(2,3,2)
sns.violinplot(x='married',y='loan_amount',hue='loan_status',d=data,palette='viridis')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,3)
sns.violinplot(x='education',y='loan_amount',hue='loan_status',d=data,palette='copper')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,4)
sns.violinplot(x='credit_history',y='loan_amount', d=data,hue='loan_status',palette='summer')

plt.subplot(2,3,5)
sns.violinplot(x='self_employed',y='loan_amount',hue='loan_status',d=data,palette='autumn')
plt.ylabel(' ')
plt.yticks([ ])

plt.subplot(2,3,6)
sns.violinplot(x='property_area', y='loan_amount',d=data,hue='loan_status',palette='PuBuGn')
plt.ylabel(' ')
plt.yticks([ ])


# In[ ]:


plt.figure(figsize=(18,5))


plt.subplot(1,3,1)
sns.distplot(data['applicant_income'],bins=30,color='r',hist_kws=dplt.figure(figsize=(18,5))


plt.subplot(1,3,1)
sns.distplot(df['applicant_income'],bins=30,color='r',hist_kws=dict(edgecolor='white'))
plt.ylabel('frequency')

plt.subplot(1,3,2)
sns.distplot(df['co-applicant_income'],bins=30,color='blue',hist_kws=dict(edgecolor='white'))

plt.subplot(1,3,3)
sns.distplot(df['loan_amount'],bins=30,color='black',hist_kws=dict(edgecolor='white'))ict(edgecolor='white'))
plt.ylabel('frequency')

plt.subplot(1,3,2)
sns.distplot(data['co-applicant_income'],bins=30,color='blue',hist_kws=dict(edgecolor='white'))

plt.subplot(1,3,3)
sns.distplot(data['loan_amount'],bins=30,color='black',hist_kws=dict(edgecolor='white'))


# In[26]:


px.scatter_3d(frame=data,x='applicant_income',y='co-applicant_income',z='loan_amount',color='loan_status')


# In[ ]:





# In[ ]:





# In[27]:


model1=LabelEncoder()


# In[28]:


model1.fit(data['gender'])


# In[29]:


data['gender']= model1.transform(data['gender'])


# In[30]:


model2=LabelEncoder()


# In[31]:


model2.fit(data['married'])


# In[32]:


data['married']= model2.transform(data['married'])


# In[33]:


model3=LabelEncoder()


# In[34]:


model3.fit(data['education'])


# In[35]:


data['education']= model3.transform(data['education'])


# In[36]:


model4=LabelEncoder()


# In[37]:


model4.fit(data['self_employed'])


# In[38]:


data['self_employed']= model4.transform(data['self_employed'])


# In[39]:


model5=LabelEncoder()


# In[40]:


model5.fit(data['property_area'])


# In[41]:


data['property_area']= model5.transform(data['property_area'])


# In[42]:


model6=LabelEncoder()


# In[43]:


model6.fit(data['loan_status'])


# In[44]:


data['loan_status']= model6.transform(data['loan_status'])


# In[45]:


data.head()


# In[46]:


plt.figure(figsize=(12,8))

corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, square=True,annot=True,linewidths=2, cmap='viridis')
plt.title('Correlation Matrix for Loan Status')


# In[52]:


X=data.drop(['loan_id','loan_status'],axis=1)
y=data['loan_status']


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# In[54]:


lr=LogisticRegression()


# In[55]:


lr.fit(X_train, y_train)


# In[56]:


lr_prediction=lr.predict(X_test)


# In[57]:


print(confusion_matrix(y_test,lr_prediction))
print('\n')
print(classification_report(y_test,lr_prediction))
print('\n')
print('Logistic Regression accuracy: ', accuracy_score(y_test,lr_prediction))


# In[58]:


dt=DecisionTreeClassifier()


# In[59]:


dt.fit(X_train, y_train)


# In[60]:


dt_prediction=dt.predict(X_test)


# In[61]:


print(confusion_matrix(y_test,dt_prediction))
print('\n')
print(classification_report(y_test,dt_prediction))
print('\n')
print('Decision Tree Accuracy: ', accuracy_score(y_test,dt_prediction))


# In[62]:


rf=RandomForestClassifier(n_estimators=200)


# In[64]:


rf.fit(X_train, y_train)


# In[65]:


rf_prediction=rf.predict(X_test)


# In[66]:


print(confusion_matrix(y_test,rf_prediction))
print('\n')
print(classification_report(y_test,rf_prediction))
print('\n')
print('Random Forest Accuracy: ', accuracy_score(y_test,rf_prediction))


# In[68]:


4-4-KNearest Neighbors


# In[69]:


error_rate=[]
for n in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    knn_prediction=knn.predict(X_test)
    error_rate.append(np.mean(knn_prediction!=y_test))
print(error_rate)


# In[70]:


plt.figure(figsize=(8,6))
sns.set_style('whitegrid')
plt.plot(list(range(1,40)),error_rate,color='b', marker='o', linewidth=2, markersize=12, markerfacecolor='r', markeredgecolor='r')
plt.xlabel('Number of Neighbors')
plt.ylabel('Error Rate')
plt.title('Elbow Method')


# In[71]:


knn=KNeighborsClassifier(n_neighbors=23)


# In[72]:


knn.fit(X_train, y_train)


# In[73]:


knn_prediction=knn.predict(X_test)


# In[74]:


print(confusion_matrix(y_test,knn_prediction))
print('\n')
print(classification_report(y_test,knn_prediction))
print('\n')
print('KNN accuracy Accuracy: ', accuracy_score(y_test,knn_prediction))


# In[ ]:


4-5-SVC


# In[75]:


svc=SVC()


# In[76]:


svc.fit(X_train, y_train)


# In[77]:


svc_prediction=svc.predict(X_test)


# In[81]:


print(confusion_matrix(y_test,svc_prediction))
print('\n')
print(classification_report(y_test,svc_prediction))
print('\n')
print('K means clustering File: ', accuracy_score(y_test,svc_prediction))


# In[87]:


print('Logistic Regression Algorithm Accuracy: ', accuracy_score(y_test,lr_prediction))
print('Decision Tree  Algorithm Accuracy : ', accuracy_score(y_test,dt_prediction))
print('Random Forest Algorithm Accuracy: ', accuracy_score(y_test,rf_prediction))
print('KNN Algorithm Accuracy: ', accuracy_score(y_test,knn_prediction))
print('K means clustering Algorithm Accuracy : ', accuracy_score(y_test,svc_prediction))


# In[ ]:




