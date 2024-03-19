#!/usr/bin/env python
# coding: utf-8

# # MULTIPLE LINEAR REGRESSION 

# ### IMPORT LIBRARIES
# 

# In[3]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
import seaborn as sns


# ### IMPORT DATASET

# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:/Users/lenovo/Downloads/combined+cycle+power+plant/CCPP')


# In[6]:


data = pd.read_excel('Folds5x2_pp.xlsx')


# In[7]:


data


# ### DEFINING X AND Y

# In[8]:


x = data.drop(columns = ['PE'], axis=1).values


# In[9]:


y = data['PE'].values


# ### SPLIT DATASET TO TRAIN AND TEST SET

# In[10]:


from sklearn.model_selection import train_test_split as tts


# In[11]:


x1,x2,y1,y2 = tts(x,y,test_size=0.33,random_state=0)


# ### TRAIN THE MODEL  

# In[12]:


from sklearn.linear_model import LinearRegression as LR


# In[13]:


reg = LR()


# In[14]:


reg.fit(x1,y1)


# ### PREDICT THE TEST SET RESULTS

# In[15]:


yp = reg.predict(x2)


# In[16]:


yp


# In[17]:


reg.predict([[25.1,62.96,1020.04,59.08]])


# In[18]:


reg.intercept_,reg.coef_


# ### EVALUATE THE MODEL  

# In[19]:


from sklearn.metrics import r2_score 


# In[20]:


r2 = r2_score(y2,yp)


# ### PLOT THE RESULTS
# 

# In[21]:


plt.figure(figsize=(15,15))
plt.scatter(y2,yp,color="pink")
plt.xlabel("Actual",size=20)
plt.ylabel("Predicted",size=20)
plt.title("Actual vs Predicted",size=30);


# ### RESIDUALS
# 
# 

# In[22]:


pred_y = pd.DataFrame({'Actual Value': y2,
                      'Predicted Value':yp,
                      'Residual': y2-yp})


# In[23]:


pred_y


# In[24]:


RSS = sum([x**2 for x in pred_y['Residual']])


# In[25]:


RSS


# In[26]:


sum([x**2 for x in range(5)])


# In[27]:


corr = data.corr()
corr


# In[28]:


sns.heatmap(corr)


# In[29]:


def correlation(df,threshold):
    corr_cols = set()
    corrmatrix = df.corr()
    for i in range(len(corrmatrix.columns)):
        for j in range(i):
            if abs(corrmatrix.iloc[i,j]) > threshold:
                colname = corrmatrix.columns[i]
                corr_cols.add(colname)
    return corr_cols


# In[30]:


correlation(data,0.7)


# In[31]:


res = pred_y['Residual']


# In[32]:


sum([x**2 for x in res])


# In[33]:


data.isna().sum()


# In[34]:


import statsmodels.api as sm


# In[35]:


xt = sm.add_constant(x1)


# In[36]:


lin = sm.OLS(y1,xt).fit()


# In[37]:


lin.params


# In[38]:


lin.summary()


# In[39]:


VIF = round(1/(1-r2**2))


# In[40]:


VIF


# In[41]:


xd = pd.DataFrame(x)
xd.describe()


# In[42]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_s = ss.fit_transform(x)

pd.DataFrame(x_s).describe()


# In[43]:


yd = pd.DataFrame(y)
y_s = ss.fit_transform(yd)
y_s.mean()


# In[44]:


xt, xtt, yt, ytt = tts(x_s,y_s,test_size=0.33,random_state=0)


# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


model2 = LinearRegression()
model2.fit(xt,yt)


# In[47]:


model2.score(xtt,ytt)


# In[53]:


model2.intercept_,model2.coef_


# In[ ]:




