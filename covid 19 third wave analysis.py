#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np


# In[4]:


dataset=pd.read_csv('covid_19_india.csv',parse_dates=['Date'],dayfirst=True)


# In[5]:


dataset.head()


# In[6]:


dataset=dataset[['Date','State/UnionTerritory','Cured','Deaths','Confirmed']]
dataset.columns=['date','state','cured','deaths','confirmed']


# In[7]:


dataset.head()


# In[9]:


dataset


# In[10]:


dataset.tail()


# In[13]:


today=dataset[dataset.date == '2021-05-16']


# In[14]:


today


# In[16]:


max_confirmed_cases=today.sort_values(by="confirmed",ascending=False)
max_confirmed_cases


# In[18]:


top_states_with_confirmed_cases=max_confirmed_cases[0:5]


# In[20]:


sns.set(rc={'figure.figsize' :(15,10)})
sns.barplot(x="state",y="confirmed",data=top_states_with_confirmed_cases,hue="state")
plt.show()


# In[21]:


max_death_cases=today.sort_values(by="deaths",ascending=False)
max_death_cases


# In[23]:


top_states_with_confirmed_deaths=max_death_cases[0:5]


# In[27]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="state",y="deaths",data=top_states_with_confirmed_deaths,hue="state")
plt.show()


# In[28]:


max_cured_cases=today.sort_values(by="cured",ascending=False)
max_cured_cases


# In[29]:


top_states_with_cured=max_cured_cases[0:5]


# In[30]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="state",y="cured",data=top_states_with_cured,hue="state")
plt.show()


# In[33]:


maha=dataset[dataset.state == 'Maharashtra']


# In[34]:


maha


# In[36]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=maha,color="g")
plt.show()


# In[37]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="deaths",data=maha,color="r")
plt.show()


# In[39]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="cured",data=maha,color="b")
plt.show()


# In[52]:


kerala = dataset[dataset.state == 'Kerala']


# In[53]:


kerala


# In[54]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=kerala,color="g")
plt.show()


# In[56]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="deaths",data=kerala,color="r")
plt.show()


# In[58]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="cured",data=kerala,color="b")
plt.show()


# In[59]:


telangana=dataset[dataset.state == 'Telangana']


# In[60]:


telangana


# In[61]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="deaths",data=telangana,color="r")
plt.show()


# In[63]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="confirmed",data=telangana,color="g")
plt.show()


# In[64]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="cured",data=telangana,color="b")
plt.show()


# In[69]:


Tests = pd.read_csv('StatewiseTestingDetails.csv')
Tests


# In[78]:


test_latest = tests[tests.Date == '2021-05-15']


# In[79]:


test_latest


# In[83]:


max_tests_State=test_latest.sort_values(by="TotalSamples",ascending=False)
max_tests_State


# In[85]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x="State",y="TotalSamples",data=max_tests_State[0:5],hue="State")
plt.show()


# In[88]:


from sklearn.model_selection import train_test_split


# In[122]:


telangana


# In[127]:


telangana['date']=telangana['date'].map(dt.datetime.toordinal)
telangana.head()


# In[128]:


x=telangana['date']
y=telangana['confirmed']


# In[125]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[129]:


from sklearn.linear_model import LinearRegression


# In[130]:


lr = LinearRegression()


# In[135]:


y_train


# In[136]:


lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))


# In[ ]:





# In[ ]:





# In[ ]:




