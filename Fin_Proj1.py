#!/usr/bin/env python
# coding: utf-8


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# # Reading the datasets

# In[2]:


train_data = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\sales_train.csv",parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)
items = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\items.csv")
shops = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\shops.csv")
item_categories = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\item_categories.csv")
test_data = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\test.csv")
sample_submission = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\sample_submission.csv")


# # Exploring the training datset

# In[3]:


train_data.head()


# Here we can see that every variable data is interlinked with other datasets.
# To read an item, we have to merge that with the other datasets which contain shop_id,shop_name,item_id and fianlly item. 

# In[4]:


train_data.describe()


# In[5]:


train_data.info()


# # Data cleaning 

# In[6]:


train_data.isnull().sum() 


# By implementing isnull() module we can see whether there are null values in the dataset.Here we can see that there are '0' null values.

# In[7]:


train_data['date'] = pd.to_datetime(train_data['date'])
print(train_data['date'])


#  Date format using "dd.mm.yyyy". We are going to change date format into pandas date format.

# In[8]:


#Change item price into positive number
train_data['item_price'] = abs(train_data['item_price'])
print(train_data['item_price'])

#Change item cnt day into integer type
train_data['item_cnt_day'] = train_data['item_cnt_day'].astype('int')
print(train_data['item_cnt_day'])

#Change item cnt day into positive number
train_data['item_cnt_day'] = abs(train_data['item_cnt_day'])
print(train_data['item_cnt_day'])


# Some item prices might have negative values. So we have to change item price into absolute value (all value become positive)
# and item cnt day has float type and some item cnt day has negative value. I will change item cnt day into integer type and make int cnt day into positive (all value become positive)

# # Exploratory Data Analysis(Understanding the Datasets)

# In[9]:


items.head(3)


# item_name, item_id, item_category_id are the categories in this datasets 

# In[10]:


shops.head()


# shop_name, shop_id are the categories in this datasets 

# In[11]:


item_categories.head(3)


# item_category_name, item_category_id are the categories in this datasets

# In[12]:


test_data.head(3)


# shop_id, item_id are the categories in this datasets

# In[13]:


sample_submission.head(3) 


# sample predicted item_cnt_month for the given datasets

# In[14]:


print("No. of Null values in the test set :", test_data.isnull().sum().sum())
print("No. of Null values in the item set :", items.isnull().sum().sum())
print("No. of Null values in the shops set :", shops.isnull().sum().sum())
print("No. of Null values in the item_categories set :", item_categories.isnull().sum().sum())


# We can observe that there are '0' null values in all the datasets

# In[15]:


plt.rcParams['figure.figsize'] = (19, 9)
sns.barplot(x=items['item_category_id'], y=items['item_id'], palette = 'colorblind')
plt.title('Count for Different Items Categories', fontsize = 20)
plt.xlabel('Item Categories', fontsize = 15)
plt.ylabel('Items in each Categories', fontsize = 15)
plt.show()


# looking at the number of different categories in all the items and count of different items in each categories

# #### Exploring Train dataset

# In[16]:


jan_count = len(train_data[train_data['date_block_num']==0])+len(train_data[train_data['date_block_num']==12])+len(train_data[train_data['date_block_num']==24])
feb_count = len(train_data[train_data['date_block_num']==1])+len(train_data[train_data['date_block_num']==13])+len(train_data[train_data['date_block_num']==25])
mar_count = len(train_data[train_data['date_block_num']==2])+len(train_data[train_data['date_block_num']==14])+len(train_data[train_data['date_block_num']==26])
apr_count = len(train_data[train_data['date_block_num']==3])+len(train_data[train_data['date_block_num']==15])+len(train_data[train_data['date_block_num']==27])
may_count = len(train_data[train_data['date_block_num']==4])+len(train_data[train_data['date_block_num']==16])+len(train_data[train_data['date_block_num']==28])
jun_count = len(train_data[train_data['date_block_num']==5])+len(train_data[train_data['date_block_num']==17])+len(train_data[train_data['date_block_num']==29])
jul_count = len(train_data[train_data['date_block_num']==6])+len(train_data[train_data['date_block_num']==18])+len(train_data[train_data['date_block_num']==30])
aug_count = len(train_data[train_data['date_block_num']==7])+len(train_data[train_data['date_block_num']==19])+len(train_data[train_data['date_block_num']==31])
sep_count = len(train_data[train_data['date_block_num']==8])+len(train_data[train_data['date_block_num']==20])+len(train_data[train_data['date_block_num']==32])
oct_count = len(train_data[train_data['date_block_num']==9])+len(train_data[train_data['date_block_num']==21])+len(train_data[train_data['date_block_num']==33])
nov_count = len(train_data[train_data['date_block_num']==10])+len(train_data[train_data['date_block_num']==22])
dec_count = len(train_data[train_data['date_block_num']==11])+len(train_data[train_data['date_block_num']==23])

#month bar chart
month_freq = pd.DataFrame({'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec'],'val': [jan_count, feb_count, mar_count, apr_count, may_count, jun_count, jul_count, aug_count, sep_count, oct_count, nov_count, dec_count] })
fig=plt.figure()
sns.barplot(x='val', y='month', data=month_freq)
plt.title("Frequency of month")
plt.show()


# Most transaction happened in January, followed by March, December, etc.

# Having a look at the distribution of item sold per day. It describes different blocks of months and number of purchases

# In[17]:


plt.rcParams['figure.figsize'] = (13, 7)
sns.distplot(train_data['item_price'], color = 'red')
plt.title('Distribution of the price of Items', fontsize = 20)
plt.xlabel('Range of price of items', fontsize = 15)
plt.ylabel('Distrbution of prices over items', fontsize = 15)
plt.show()


# Having a look at the distribution of item price, it describes about range of prices of items and distribution of the price of items

# In[18]:


plt.rcParams['figure.figsize'] = (13, 7)
sns.distplot(train_data['item_cnt_day'], color = 'purple')
plt.title('Distribution of the no. of Items Sold per Day', fontsize = 20)
plt.xlabel('Range of items sold per day', fontsize = 15)
plt.ylabel('Distrbutions per day', fontsize = 15)
plt.show()


#  having a look at the distribution of item sold per day and the range of items sold per day to distribution per day

# In[19]:


x = train_data['item_id'].nunique()
print("The No. of Unique Items Present in the stores available: ", x)

x = item_categories['item_category_id'].nunique()
print("The No. of Unique categories for Items Present in the stores available: ", x)

x = train_data['shop_id'].nunique()
print("No. of Unique Shops are :", x)


# Checking the no. of unique item present in the stores and 
# checking the no. of unique shops given in the dataset
# 
# 

# In[20]:


# making a new column day
train_data['day'] = train_data['date'].dt.day

# making a new column month
train_data['month'] = train_data['date'].dt.month

# making a new column year
train_data['year'] = train_data['date'].dt.year

# making a new column week
train_data['week'] = train_data['date'].dt.week

# checking the new columns
train_data.columns
print(train_data.head(3))


# In[21]:


# checking which months are most busiest for the shops

plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(train_data['month'], palette = 'dark')
plt.title('The most busiest months for the shops', fontsize = 20)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()


# ![image.png](attachment:image.png)

# Here we can observe the busiest months for the shops in a year.
# We can obsereve the frequency of the months sales. '1' intends to January where it reaches the highest frequency of items in a shop 

# In[22]:


# checking the columns of the train data

train_data.columns


# # Feature engineering

# In[23]:


train_data['revenue'] = train_data['item_price'] * train_data['item_cnt_day']
print(train_data.head(5))

sns.distplot(train_data['revenue'], color = 'blue')
plt.title('Distribution of Revenue', fontsize = 20)
plt.xlabel('Range of Revenue', fontsize = 10)
plt.ylabel('Revenue')
plt.show()


# Here we added a new column which gives us the information about the revenue generated for a shop and how it varies in the range

# In[24]:


# plotting a box plot for itemprice and item-cnt-day

plt.rcParams['figure.figsize'] = (15, 7)
sns.violinplot(x = train_data['day'], y = train_data['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# Here we can see the box plot for revenue and days in a month

# In[25]:


sales_shop = train_data.groupby('shop_id')['revenue'].sum().sort_values(ascending=False).reset_index().head(10)
fig=plt.figure() 
sns.barplot(y='revenue', x='shop_id', data=sales_shop)
plt.title("Top 10 shops with highest sales")
plt.ylabel('Total Sales')
plt.xlabel("Shop id")
plt.show()


# Shop with id 31 has highest total sales, followed by shop with id 25, etc.

# In[26]:


sales_item = train_data.groupby('item_id')['revenue'].sum().sort_values(ascending=False).reset_index().head(10)
fig=plt.figure() 
sns.barplot(y='revenue', x='item_id', data=sales_item)
plt.title("Top 10 items with highest sales")
plt.ylabel('Total Sales')
plt.xlabel("Item id")
plt.show()


# Item with id 6675 has highest total sales, followed by item with id 3732, etc.

# In[27]:


# converting the data into monthly sales data

# making a dataset with only monthly sales data
data = train_data.groupby([train_data['date'].apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()

# specifying the important attributes which we want to add to the data
data = data[['date','item_id','shop_id','item_cnt_day']]

# At last we can select the specific attributes from the dataset which are important 
data = data.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_day', fill_value = 0).reset_index()

# looking at the newly prepared datset
data.shape
print(data)


# # Modelling 

# In[28]:


# let's merge the monthly sales data prepared to the test data set

test_data = pd.merge(test_data, data, on = ['item_id', 'shop_id'], how = 'left')

# filling the empty values found in the dataset
test_data.fillna(0, inplace = True)

# checking the dataset
test_data.head()


# In[29]:


# now let's create the actual training data

x_train = test_data.drop(['2015-10', 'item_id', 'shop_id'], axis = 1)
print(x_train.head())
y_train = test_data['2015-10']
print(y_train.head())

# deleting the first column so that it can predict the future sales data
x_test = test_data.drop(['2013-01', 'item_id', 'shop_id'], axis = 1)
print(x_test.head())

# checking the shapes of the datasets
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_train.shape)


# In[30]:


from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)


# Creating model
from lightgbm import LGBMRegressor

model = LGBMRegressor( n_estimators=200,
                           learning_rate=0.03,
                           num_leaves=32,
                           colsample_bytree=0.9497036,
                           subsample=0.8715623,
                           max_depth=8,
                           reg_alpha=0.04,
                           reg_lambda=0.073,
                           min_split_gain=0.0222415,
                           min_child_weight=40)
model.fit(x_train, y_train)
print(model)


# In[31]:


pred = model.predict(x_valid)
print(pred)
predictions = [round(value) for value in pred]


# In[32]:


from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_valid, predictions)
print(type(y_valid))
print(type(predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Here we predicted the test dataset values.

# In[33]:


# Get the test set predictions and clip values to the specified range
y_pred = model.predict(x_test).clip(0., 20.)

# Create the submission file and submit
future_sales_prediction = pd.DataFrame(y_pred, columns=['item_cnt_month'])
future_sales_prediction.to_csv('future predictions sales .csv',index_label='ID')
print(future_sales_prediction)


# Finally our task which is to forecast the total amount of products sold in every shop for the test set is completed.
# Created a robust model that can handle such situations.
# 
# 

# In[34]:


mse_f = np.mean(y_pred**2)
mae_f = np.mean(abs(y_pred)) #mean absolute error
rmse_f = np.sqrt(mse_f)
r2_f = 1-(sum(y_pred**2)/sum((y_pred-np.mean(y_pred))**2))

print("Results by manual calculation:")
print("MAE:",mae_f)
print("MSE:", mse_f)
print("RMSE:", rmse_f)
print("R-Squared:", r2_f)


# #### This result is 0.8637265595860395 accurate

# # Implementing with another Model

# In[35]:


test_data1 = pd.read_csv(r"E:\Masters\CS636\competitive-data-science-predict-future-sales\test.csv")


# In[36]:


# converting the data into monthly sales data

# making a dataset with only monthly sales data
data = train_data.groupby([train_data['date'].apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()

# specifying the important attributes which we want to add to the data
data = data[['date','item_id','shop_id','item_cnt_day']]

# At last we can select the specific attributes from the dataset which are important 
data = data.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_day', fill_value = 0).reset_index()

# looking at the newly prepared datset
data.shape
print(data)


# In[37]:


test_data1 = pd.merge(test_data1, data, on = ['item_id', 'shop_id'], how = 'left')

# filling the empty values found in the dataset
test_data1.fillna(0, inplace = True)


# In[38]:


x_train = test_data1.drop(['2015-10', 'item_id', 'shop_id'], axis = 1)
y_train = test_data1['2015-10']

#dividing the train data into x_train and y_train and featuring test data


# # Building the model

# In[39]:


from xgboost import XGBRegressor
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
model= XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(x_train, y_train)
testpred = model.predict(x_valid)
predictions = [round(value) for value in testpred]


# In[40]:


#Accuracy of the model
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_valid, predictions)
print(type(y_valid))
print(type(predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[41]:


Xgb_prediction=model.predict(x_test)
Xgb_prediction


# In[42]:


Prediction_xgb = pd.DataFrame(Xgb_prediction, columns = ['item_cnt_month'])
Prediction_xgb['ID'] = Prediction_xgb.index
Prediction_xgb = Prediction_xgb.set_index('ID')
Prediction_xgb.to_csv('pred_xgb.csv')


# In[43]:


Prediction_xgb


# Finally our task which is to forecast the total amount of products sold in every shop for the test set is completed.
# Created a robust model that can handle such situations.
# 
# 

# In[44]:


#Prediction accuracy
mse_ = np.mean(Prediction_xgb**2)
r2_ = 1-(sum(Xgb_prediction**2)/sum((Xgb_prediction-np.mean(Xgb_prediction))**2))

print("Results by manual calculation:")
print("MAE:",mse_)
print("R-Squared:", r2_)


# #### The results are 0.8908540854796994 accurate

# # Comparing the both models Xgb and lgbm regression

# #### XGBoost
# XGBoost (eXtreme Gradient Boosting) is a machine learning algorithm that focuses on computation speed and model performance. It was introduced by Tianqi Chen and is currently a part of a wider toolkit by DMLC (Distributed Machine Learning Community). The algorithm can be used for both regression and classification tasks and has been designed to work with large and complicated datasets.
# 
# 
# Source
# The model supports the following kinds of boosting:
# 
# Gradient Boosting as controlled by the learning rate
# Stochastic Gradient Boosting that leverages sub-sampling at a row, column or column per split levels
# Regularized Gradient Boosting using L1 (Lasso) and L2 (Ridge) regularization 
# Some of the other features that are offered from a system performance point of view are:
# 
# Using a cluster of machines to train a model using distributed computing
# Utilization of all the available cores of a CPU during tree construction for parallelization
# Out-of-core computing when working with datasets that do not fit into memory
# Making the best use of hardware with cache optimization

# #### LightGBM
# Similar to XGBoost, LightGBM (by Microsoft) is a distributed high-performance framework that uses decision trees for ranking, classification, and regression tasks.
# 
# 
# Source
# The advantages are as follows:
# 
# Faster training speed and accuracy resulting from LightGBM being a histogram-based algorithm that performs bucketing of values (also requires lesser memory)
# Also compatible with large and complex datasets but is much faster during training
# Support for both parallel learning and GPU learning
# In contrast to the level-wise (horizontal) growth in XGBoost, LightGBM carries out leaf-wise (vertical) growth that results in more loss reduction and in turn higher accuracy while being faster. But this may also result in overfitting on the training data which could be handled using the max-depth parameter that specifies where the splitting would occur. Hence, XGBoost is capable of building more robust models than LightGBM.

# First we implemented lgbm model which we got a good accuarcy for the prediction and to increase the accuracy we did another machine learning 
# model 'xgboost'.
# In our view Xgboost model performed with more accuaracy but it do consists if some complex data featuring.
# 
# Accuracy for lgbm = 0.8655043586550436     .. 
# Accuracy for xgboost = 0.89012567837444

# ### Thank You

# In[ ]:




