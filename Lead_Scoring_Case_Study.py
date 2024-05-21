#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study

# In[1]:


#import the warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#import the libs for analysis, data handling
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)


# # Step 1: Importing the dataset

# In[3]:


leads = pd.read_csv(r"C:\Users\shaik\OneDrive\Downloads\Lead_Scoring_Case_Study_ML\proj\Leads.csv")


# In[4]:


leads.head()


# # Step 2: Inspecting the Dataset

# In[5]:


leads.shape


# In[6]:


leads


# In[7]:


leads.info()


# In[8]:


leads.describe()


# In[9]:


leads.isnull().sum()


# In[10]:


leads.isnull().sum()*100/leads.shape[0]


# # Step 3: Data Cleaning

# In[11]:


#Drop the Prospect ID & Lead Number
leads.drop(columns=["Prospect ID", "Lead Number"], axis=1, inplace=True)


# In[12]:


#list the categorical columns
cat_cols = ["Lead Origin", "Lead Source", "Country", "Specialization", "How did you hear about X Education", 
            "What is your current occupation", "What matters most to you in choosing a course", "Search",
            "Magazine", "Newspaper Article", "X Education Forums", "Newspaper", "Digital Advertisement",
            "Tags", "Lead Quality","Lead Profile", "City", "Last Notable Activity"]


# In[13]:


cat_cols


# In[14]:


#Check the unique values in each category
for i in cat_cols:
    print(i, ":", leads[i].value_counts())
    print("----------------------------------------------------------------")


# In[15]:


leads.isnull().sum()*100/leads.shape[0]


# # Impute "Select" value as null

# In[16]:


sel = ["Specialization", "How did you hear about X Education", "Lead Profile", "City"]
sel


# In[17]:


leads[sel] = leads[sel].replace("Select", np.nan)


# In[18]:


leads.isnull().sum()*100/leads.shape[0]


# # Dropping columns with null values greater than 40%

# In[19]:


cols = leads.columns
for i in cols:
    if leads[i].isnull().sum()*100/leads.shape[0] > 40:
        leads.drop(i, axis=1, inplace = True)


# In[20]:


leads.isnull().sum()*100/leads.shape[0]


# In[21]:


cols = leads.columns
for i in cols:
    if leads[i].isnull().sum()*100/leads.shape[0] < 15 and leads[i].dtype == 'object':
        leads[i].replace(np.nan, leads[i].mode, inplace=True)


# In[22]:


leads.isnull().sum()*100/leads.shape[0]


# In[23]:


#Imputing missing values >15 and <40 to "other" category
cols = leads.columns
for i in cols:
    if leads[i].isnull().sum()*100/leads.shape[0] > 15 and leads[i].dtype == 'object':
        leads[i].replace(np.nan, "Others", inplace=True)


# In[24]:


leads.isnull().sum()*100/leads.shape[0]


# In[25]:


leads.shape


# In[26]:


leads.info()


# In[27]:


#Exclude the roots with null values on columns TotalVisits & Page View Per Visit
leads = leads[~pd.isnull(leads["TotalVisits"])]
leads = leads[~pd.isnull(leads["Page Views Per Visit"])]


# In[28]:


leads.info()


# In[29]:


leads.isnull().sum()*100/leads.shape[0]


# In[30]:


leads.isnull().sum()


# # Step 4: EDA on the Dataset

# In[31]:


#import the visualization libs
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


cat_obj = list(leads.select_dtypes(include='object'))
cat_obj


# In[33]:


for i in cat_obj:
    print(i)
    plt.figure(figsize=[30,10])
    sns.countplot(x=i, hue="Converted", data=leads)
    plt.show()


# ### Insights
# Leads with Do not call and do not email are not likely to take up the course
# 
# Last Activity with email opened and SMS send are likely to take up the course
# 
# Majority of the leads are from India
# 
# Most people with other or unknown speacialization more likely to take up the course followed by finance management and Resource maketing
# 
# Working prefessionals more likely to take up the course

# In[34]:


#pair plot on numerical values
cat_num = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
sns.pairplot(leads[cat_num])
plt.show()


# In[35]:


#Correlation in Numerical Columns
num = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit", "Converted"]
plt.figure(figsize=[10,10])
sns.heatmap(leads[num].corr(),
           annot=True, cmap="BuPu")


# ### Insights
# Total Visit and Page View Per Visit are positively correlated with a values of 0.51
# 
# Total Time spend on website has positive correlation with the Page Views per Visit with a value of 0.32

# In[36]:


binary_cols = ["Do Not Email", "Do Not Call", "Through Recommendations", "Receive More Updates About Our Courses",
              "Update me on Supply Chain Content", "I agree to pay the amount through cheque",
              "A free copy of Mastering The Interview", "Search", "Magazine", "Newspaper Article",
               "Newspaper", "X Education Forums", "Digital Advertisement", "Get updates on DM Content"]
leads[binary_cols] = leads[binary_cols].apply(lambda x: x.map({"Yes":1, "No" :0}))
leads.head()


# # Data Preparation
# Create Dummies for categorical variables
# 
# Perform Train Test Split
# 
# Perform Scaling

# In[37]:


cat_cols = list(leads.select_dtypes(include='object'))
cat_cols


# In[38]:


# 'Lead Origin'
a = pd.get_dummies(leads['Lead Origin'], prefix='Lead Origin', drop_first=True, dtype = int)
leads_final0 = pd.concat([leads,a], axis=1)

# 'Lead Source'
b= pd.get_dummies(leads['Lead Source'], prefix='Lead Source', drop_first=True, dtype = int)
leads_final1 = pd.concat([leads_final0,b], axis=1)

# 'Last Activity'
c= pd.get_dummies(leads['Last Activity'], prefix='Last Activity', drop_first=True, dtype = int)
leads_final2 = pd.concat([leads_final1,c], axis=1)

# 'Country'
d= pd.get_dummies(leads['Country'], prefix='Country', drop_first=True, dtype = int)
leads_final3 = pd.concat([leads_final2,d], axis=1)

# 'Specialization'
e= pd.get_dummies(leads['Specialization'], prefix='Specialization', drop_first=True, dtype = int)
leads_final4 = pd.concat([leads_final3,e], axis=1)

# 'What is your current occupation'
f= pd.get_dummies(leads['What is your current occupation'], prefix='What is your current occupation', drop_first=True, dtype = int)
leads_final5 = pd.concat([leads_final4,f], axis=1)

# 'What matters most to you in choosing a course'
g= pd.get_dummies(leads['What matters most to you in choosing a course'], 
                  prefix='What matters most to you in choosing a course', drop_first=True, dtype = int)
leads_final6 = pd.concat([leads_final5,g], axis=1)

# 'Tags'
h= pd.get_dummies(leads['Tags'], prefix='Tags', drop_first=True, dtype = int)
leads_final7 = pd.concat([leads_final6,h], axis=1)

# 'City'
i= pd.get_dummies(leads['City'], prefix='City', drop_first=True, dtype = int)
leads_final8 = pd.concat([leads_final7,i], axis=1)

# 'Last Notable Activity'
j= pd.get_dummies(leads['Last Notable Activity'], prefix='Last Notable Activity', drop_first=True, dtype = int)
leads_final = pd.concat([leads_final8,j], axis=1)


# In[39]:


leads_final.head()


# In[40]:


leads_final.drop(columns=cat_cols, axis=1, inplace=True)


# In[41]:


leads_final.info(verbose=True)


# In[42]:


leads_final.head()


# In[43]:


#Target and feature variables split
x = leads_final.drop("Converted", axis=1)
y = leads_final["Converted"]


# In[44]:


#Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=100)


# In[46]:


#Scaling the data
scaler = StandardScaler()
X_train[["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]] =  scaler.fit_transform(
    X_train[["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]])


# In[47]:


X_train.head()


# In[48]:


converted = sum(leads['Converted'])/len(leads['Converted'].index)
converted


# # Step 6: Model Building

# In[49]:


#import the ML libraries
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


# In[50]:


X_train.shape


# In[51]:


#Top 15 feature selection by using RFE
logregg = LogisticRegression()
rfe = RFE(logregg, n_features_to_select=15)
rfe.fit(X_train, Y_train)


# In[52]:


rfe.support_


# In[53]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[54]:


col = X_train.columns[rfe.support_]
col


# In[55]:


X_train.columns[~rfe.support_]


# # Model 1

# ## Assessing the model with Statsmodels

# In[56]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[57]:


Y_train_pred = res.predict(X_train_sm)
Y_train_pred[:10]


# In[58]:


Y_train_pred = Y_train_pred.values.reshape(-1)
Y_train_pred[:10]


# In[59]:


Y_train_pred_final = pd.DataFrame({'Converted_val':Y_train.values, 'Converted_Prob':Y_train_pred})
Y_train_pred_final.head()


# In[60]:


Y_train_pred_final['predicted'] = Y_train_pred_final["Converted_Prob"].map(lambda x: 1 if x > 0.5 else 0)
Y_train_pred_final.head()


# In[61]:


cm = metrics.confusion_matrix(Y_train_pred_final["Converted_val"], Y_train_pred_final["predicted"])
cm


# In[62]:


acc = metrics.accuracy_score(Y_train_pred_final["Converted_val"], Y_train_pred_final["predicted"])
acc


# # Changing VIFs

# In[63]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending = False)


# In[64]:


vif


# In[65]:


col = col.drop('Tags_Interested in Next batch',i)
col


# # Model 2

# In[66]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[67]:


vif1 = pd.DataFrame()
vif1['Features'] = X_train[col].columns
vif1['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif1['VIF'] = round(vif1['VIF'], 2)
vif1 = vif1.sort_values(by="VIF", ascending = False)


# In[68]:


vif1


# In[69]:


col = col.drop('Tags_Lateral student',i)
col


# # Model 3

# In[70]:


X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[71]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending = False)


# In[72]:


vif


# In[73]:


col = col.drop('Tags_wrong number given',i)
col = col.drop('Tags_invalid number',i)
col


# # Model 4

# In[74]:


X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[75]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending = False)


# In[76]:


vif


# In[77]:


Y_train_pred = res.predict(X_train_sm)
Y_train_pred[:10]


# In[78]:


Y_train_pred = Y_train_pred.values.reshape(-1)
Y_train_pred[:10]


# In[79]:


Y_train_pred_final = pd.DataFrame({'Converted_val':Y_train.values, 'Converted_Prob':Y_train_pred})
Y_train_pred_final.head()


# In[80]:


Y_train_pred_final['predicted'] = Y_train_pred_final["Converted_Prob"].map(lambda x: 1 if x > 0.5 else 0)
Y_train_pred_final.head()


# In[81]:


cm = metrics.confusion_matrix(Y_train_pred_final["Converted_val"], Y_train_pred_final["predicted"])
cm


# In[82]:


acc = metrics.accuracy_score(Y_train_pred_final["Converted_val"], Y_train_pred_final["predicted"])
acc


# In[83]:


TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]


# In[84]:


TP / float(TP+FN)


# In[85]:


TN / float(TN+FP)


# In[86]:


FP / float(TN+FP)


# In[87]:


TP / float(TP+FP)


# In[88]:


TN / float(TN+FN)


# In[ ]:




