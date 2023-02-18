#!/usr/bin/env python
# coding: utf-8

# # Optimizing E-commerce Logistics

# ## 1. Problem definition

# This research aims to develop a neural network model that can distinguish between on-time and delay shipments, then we are going implement our model on test set to forecast whether these shipments will arrive on-time or not.

# ## 2. Data Preparation

# * Data source: The dataset, collected by an international e-commerce company, records all shipment details [Link](https://www.kaggle.com/datasets/prachi13/customer-analytics).
# * Data organization: 1 CSV file organized in a long data format.
# * Sample size: 10,999 observations.
# * Number of features: 12 columns.

# ## 3. Data Preprocessing

# ### a. Data cleaning

# First we import required packages to construct Neural Network model and Outcome Summary:

# In[1]:


#Regular EDA (exploratory data analysis) and plotting libraries
import math #basic calculation
import pandas as pd
import seaborn as sns #plot in data analysis
import numpy as np
import matplotlib.pylab as plt #plot confusion matrix
import matplotlib.pyplot as plt #plot confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')

#Package for neural network
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#Package for data normalization
from sklearn.preprocessing import MinMaxScaler

#Package for model evaluation
from dmba import classificationSummary
from sklearn.metrics import confusion_matrix, accuracy_score


# In[2]:


# Loading shipments dataset
shipments = pd.read_csv('shipments.csv')


# In[3]:


# Viewing dataframe structure
shipments.shape


# In[4]:


# Running the dataset
shipments


# In[5]:


# Checking datatype
shipments.info()


# There are 8 numerical variables and 4 string variables in the original dataset

# In[6]:


#Plotting null values in our dataset by using heatmap
sns.heatmap(shipments.isnull())
plt.title("Empty Data")


# There is no missing value in our dataset

# In[7]:


#Droping ID column (because ID = index + 1)
shipments.drop(['ID'], axis=1, inplace=True)


# In[8]:


#Renaming column Reached.on.Time_Y.N (ontime = 0, delayed = 1)
shipments.rename(columns={'Reached.on.Time_Y.N': 'Shipment_status'}, inplace=True)


# In[9]:


#Running the dataset
shipments


# In[10]:


#Investigating all the elements whithin each feature 

for column in shipments: #create a loop to go through all columns in our dataset
    unique_values = np.unique(shipments[column]) #take out the unique values
    nr_values = len(unique_values) #number of unique values
    if nr_values <= 10: #if clause to print the outcomes
        print("The number of values for feature {} is: {} -- {}".format(column, nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))


# ### b. Exploratory data analysis

# In[11]:


#Investigating the distribution of outcome variable Shipment_status (ontime = 0, delayed = 1)
sns.countplot(x = 'Shipment_status', data = shipments, palette = 'Set1')


# In[12]:


#Pivot table Shipment_status with warehouse_block
pd.crosstab(shipments.Warehouse_block, shipments.Shipment_status)


# In[13]:


#Pivot table Shipment_status with Mode_of_Shipment
pd.crosstab(shipments.Mode_of_Shipment, shipments.Shipment_status)


# In[14]:


#Pivot table Shipment_status with Product_importance
pd.crosstab(shipments.Product_importance, shipments.Shipment_status)


# ### c. Data conversion

# In this section, we are going to convert categorical variables into dummy variables to fit into the neural network model

# In[15]:


#Converting categorical variables into numeric variables
processed = pd.get_dummies(shipments, columns=['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender'])


# In[16]:


#Determining outcome and predictors
outcome = 'Shipment_status'
predictors = [c for c in processed.columns if c != outcome]


# In[17]:


#Conducting data normalization
scaler = MinMaxScaler()
processed = pd.DataFrame(scaler.fit_transform(processed),index=processed.index,columns=processed.columns)


# In[18]:


processed


# In[19]:


#Converting data to binary
list_int = ['Shipment_status',
       'Warehouse_block_A', 'Warehouse_block_B', 'Warehouse_block_C',
       'Warehouse_block_D', 'Warehouse_block_F', 'Mode_of_Shipment_Flight',
       'Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship',
       'Product_importance_high', 'Product_importance_low',
       'Product_importance_medium', 'Gender_F', 'Gender_M']
processed[list_int] = processed[list_int].astype('int')


# In[20]:


processed.info()


# Outcome: We successfully scale variables to 0-1 and all categorical variables are transformed to dummy values 0-1 range

# ## 4. Data Modelling

# First, we will be specifying the network architecture, this includes:
# * Number of inputs: equal to number of predictors 19
# * Number of output nodes: our neural network is a binary classifier, then it also has a single output node
# * Number of hidden layers: most popular – we try one hidden layer for our problem
# * Number of nodes in the hidden layer: midway between input and output nodes (from 1 to 19) and equal to 2/3 input nodes + output nodes, so we are going to choose 14 as number of hidden nodes

# In[21]:


# Splitting the dataset into training set, valid set, and test set, size = 0.8:0.19:0.01
X = processed[predictors]
y = processed[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=1)
valid_X, test_X, valid_y, test_y = train_test_split(valid_X, valid_y, test_size=0.005, random_state=1)


# In[22]:


train_X.shape, valid_X.shape, test_X.shape, train_y.shape, valid_y.shape, test_y.shape


# In[23]:


# Fitting into a neural network with 1 hidden layer, 14 hidden nodes, activation function logistic
clf = MLPClassifier(hidden_layer_sizes=(14,), activation='logistic', solver='lbfgs', random_state=1)
clf.fit(train_X, train_y.values)


# In[24]:


# training performance
classificationSummary(train_y, clf.predict(train_X))

# validation performance
classificationSummary(valid_y, clf.predict(valid_X))


# In[25]:


# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':45})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Prediction')


# In[26]:


# Plotting Confusion Matrix on validation data 
cm = confusion_matrix(valid_y, clf.predict(valid_X))
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm_norm, classes = clf.classes_, title='Confusion matrix')


# ## 5. Model Implementation

# In this section we are going to use the neural network model to predict 11 records of test set

# In[27]:


test_X


# In[28]:


clf.predict(test_X)


# Outcome: 'Delayed', 'On time', 'On time', 'Delayed', 'Delayed', 'On time', 'Delayed', 'Delayed', 'Delayed', 'Delayed', 'On time'

# ## 6. Conclusion

# Model accuracy on the validation set is 0.6469, acceptable in this supply chain field. We can use this neural network model to forecast shipment status before delivery. If the outcome of prediction is 'Delayed' like shipments 1,4,5,7,8,9,10 in the test set of model implementation, we can adjust Mode_of_Shipment to ensure shipments will arrive on time, thus enhancing the performance of delivery service of the company.

# Suggestions to develop model:
# * Collect data of order date to estimate the maximum leadtime of shipment as delivery policy
# * Collect customer location data to calculate the delivery time for each ship mode
# * Feed these variables into the model for prediction and/or build new model to automate picking ship mode task when improving delivery service of the company
