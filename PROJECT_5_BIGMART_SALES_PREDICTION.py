#!/usr/bin/env python
# coding: utf-8

# # IMPORTING IMPORTANT LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , OrdinalEncoder , StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib.inline', '')


# In[2]:


# importing the dataset
data=pd.read_csv('bigmartsales.csv')


# In[3]:


data.head(50)


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


# checking the unique values
data.nunique()


# In[8]:


# replacing none values from item weight with mean of the weights
# replace Nan with mean
data.loc[:, 'Item_Weight'].replace([np.nan], [data['Item_Weight'].mean()], inplace=True)
data.head(50)


# In[9]:


# replace zeros with mean
data.loc[:, 'Item_Visibility'].replace([0], [data['Item_Visibility'].mean()], inplace=True)
data


# In[10]:


outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
outlet_size_mode


# In[11]:


miss_bool = data['Outlet_Size'].isnull()
data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[12]:


data['Outlet_Size'].isnull().sum()


# In[13]:


data.isnull().sum()


# In[14]:


data


# In[15]:


# combine item fat content
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
data['Item_Fat_Content'].value_counts()


# In[16]:


data


# # CREATING NEW VARIABLE

# In[17]:


data['New_item_type'] = data['Item_Identifier'].apply(lambda x: x[:2])
data['New_item_type']


# In[ ]:





# In[18]:


data['New_item_type']=data['New_item_type'].map({'FD':'FOOD','DR':'DRINKS','NC':'NON_CONSUMABLE'})
data['New_item_type']


# In[19]:


data


# In[20]:


data.loc[data['New_item_type']=='NON_CONSUMABLE','Item_Fat_Content']='NON_EDIBLE'


# In[21]:


data


# In[22]:


# Creating small values  for outlet establishment  year

data['Outlet_Years']=2013-data['Outlet_Establishment_Year']


# In[23]:


print(data)


# In[25]:


data


# In[26]:


data.isnull().sum()


# # EDA

# In[30]:


sns.displot(data['Item_Weight'])
plt.show()


# In[31]:


sns.displot(data['Item_Visibility'])
plt.show()


# In[32]:


sns.displot(data['Item_MRP'])
plt.show()


# In[33]:


sns.displot(data['Item_Outlet_Sales'])
plt.show()


# In[34]:


# Log transformation 
data['Item_Outlet_Sales']=np.log(data['Item_Outlet_Sales'])
sns.displot(data['Item_Outlet_Sales'])
plt.show()


# In[37]:


sns.countplot(data['Item_Fat_Content'])
plt.show()


# In[40]:


l = list(data['Item_Type'].unique())
chart = sns.countplot(data["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)
plt.show()


# In[41]:


sns.countplot(data['Outlet_Establishment_Year'])
plt.show()


# In[ ]:





# In[44]:


sns.countplot(data['Outlet_Size'])
plt.show()


# In[ ]:





# In[45]:


sns.countplot(data['Outlet_Location_Type'])
plt.show()


# In[46]:


L = list(data['Outlet_Type'].unique())
chart_ = sns.countplot(data["Outlet_Type"])
chart_.set_xticklabels(labels=L, rotation=90)
plt.show()


# # CORRELATION MATRIX

# In[48]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# # SPLITTING THE DATA INTO TARGET AND FEATURES

# In[49]:


data.columns


# In[50]:


X=data[['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'New_item_type', 'Outlet_Years']]
y=data['Item_Outlet_Sales']


# In[52]:


y


# # TRAIN TEST SPLIT

# In[53]:


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=.20,random_state=22)


# In[54]:


# splitting data into categoricl and numerical data
X_train_cat=X_train.select_dtypes(include='object')
X_train_num=X_train.select_dtypes(include=['int32','int64','float32','float64'])


# In[55]:


X_test_cat=X_test.select_dtypes(include='object')
X_test_num=X_test.select_dtypes(include=['int32','int64','float32','float64'])


# In[57]:


# data pre processing on categorical and numerical data
OE=OrdinalEncoder()
OE.fit(X_train_cat)
X_train_cat_enc=OE.transform(X_train_cat)


# In[58]:


OE=OrdinalEncoder()
OE.fit(X_test_cat)
X_test_cat_enc=OE.transform(X_test_cat)


# In[61]:


X_train_num.reset_index(drop=True,inplace=True)


# In[63]:


X_test_num.reset_index(drop=True,inplace=True)


# In[65]:


# concat the train and test data num and categorical data
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)
X_train_final=pd.concat([X_train_cat_enc_df,X_train_num],axis=1)


# In[66]:


X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)
X_test_final=pd.concat([X_test_cat_enc_df,X_test_num],axis=1)


# # MODEL BUILDING

# In[69]:


model= LinearRegression()
model.fit(X_train_final,y_train)
y_pred=model.predict(X_test_final)


# In[71]:


MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)


# In[72]:


print(MAE)
print(MSE)


# In[75]:


from sklearn.linear_model import Ridge , Lasso


# In[76]:


model_r = Ridge()
model_r.fit(X_train_final,y_train)
y_pred_r=model_r.predict(X_test_final)


# In[77]:


MAE_r=mean_absolute_error(y_test,y_pred_r)
MSE_r=mean_squared_error(y_test,y_pred_r)
print(MAE_r)
print(MSE_r)


# In[78]:


model_l = Lasso()
model_l.fit(X_train_final,y_train)
y_pred_l=model_l.predict(X_test_final)


# In[79]:


MAE_l=mean_absolute_error(y_test,y_pred_l)
MSE_l=mean_squared_error(y_test,y_pred_l)
print(MAE_l)
print(MSE_l)


# In[81]:


from sklearn.tree import DecisionTreeRegressor
model_t=DecisionTreeRegressor()
model_t.fit(X_train_final,y_train)
y_pred_t=model_t.predict(X_test_final)


# In[82]:


MAE_t=mean_absolute_error(y_test,y_pred_t)
MSE_t=mean_squared_error(y_test,y_pred_t)
print(MAE_t)
print(MSE_t)


# In[83]:


from sklearn.ensemble import RandomForestRegressor
model_R=RandomForestRegressor()
model_R.fit(X_train_final,y_train)
y_pred_R=model_R.predict(X_test_final)


# In[84]:


MAE_R=mean_absolute_error(y_test,y_pred_R)
MSE_R=mean_squared_error(y_test,y_pred_R)
print(MAE_R)
print(MSE_R)


# In[ ]:




