#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


pd.set_option('display.max_columns',999)
pd.set_option('display.max_rows', 999)
pd.set_option('expand_frame_repr', True)

pd.set_option('max_rows',999)
pd.set_option('large_repr','truncate')
pd.set_option('max_colwidth',1000)


# In[3]:


df = pd.read_csv('obesity.csv')
df2 = pd.read_csv('IndicatorNames.csv')


# In[ ]:





# In[4]:


for i in range(len(df2)):
    if str(df2['IndicatorID'][i]) in df.columns:
        df.rename(columns={str(df2['IndicatorID'][i]):str(df2['IndicatorName'][i])},inplace=True)
        #renaming columns that have df2's ID to df2's Indicator Names






# In[5]:


df3 = pd.read_csv('UARegions.csv')


# In[6]:


df['Region'] = ' '
for i in range(len(df)):
    for a in range(len(df3)):

        if df['UA'][i] == df3['UA'][a]:
            df['Region'][i] = df3['Region'][a]

#for every value in df, search through df3 and replace any that match up with the correct information



# In[ ]:





# In[7]:


for row in range(len(df)):
    if df['obesity'][row] == -1:
        df['obesity'][row] = np.nan

#replacing -1's with NaNs



# In[ ]:





# In[8]:


del df['% population aged under 18']


# In[9]:


del df['% population aged 65+']


# In[10]:


del df ['Emergency hospital admissions due to falls in people aged 65 and over']


# In[11]:


del df['Air pollution: fine particulate matter']


# In[12]:


df = pd.get_dummies(df, columns=['Region','UA'],drop_first = True) #turn words to 'binary'


# In[13]:


nan_rows = df[df.isnull().any(1)]


# In[14]:


filtered_df = df[df.notnull().all(1)]


# In[15]:


filtered_df


# In[16]:


nan_rows_copy = nan_rows.copy()


# In[17]:


filtered_df_copy = filtered_df.copy()


# In[ ]:





# In[ ]:





# In[18]:


filtered_df.describe()



# In[19]:


corr_matrix = filtered_df.corr()
corr_matrix['obesity'].sort_values(ascending=False)


# In[20]:


filtered_df.isnull().any()


# In[21]:


corr_matrix2 = nan_rows.corr()
corr_matrix2['obesity'].sort_values(ascending=False)


# In[ ]:





# In[ ]:





# In[22]:


del nan_rows['ID']
del filtered_df['ID']


# In[23]:


filtered_df.columns


# In[24]:


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
filtered_df[['Year','popCount','Pupil absence','Violent crime - violence offences per 1,000 population','Fuel poverty','Excess winter deaths index','Violent crime - sexual offences per 1,000 population','Economic inactivity rate','Affordability of home ownership','Gender pay gap (by workplace location)','Average weekly earnings']] = minmax.fit_transform(filtered_df[['Year','popCount','Pupil absence','Violent crime - violence offences per 1,000 population','Fuel poverty','Excess winter deaths index','Violent crime - sexual offences per 1,000 population','Economic inactivity rate','Affordability of home ownership','Gender pay gap (by workplace location)','Average weekly earnings']])
nan_rows[['Year','popCount','Pupil absence','Violent crime - violence offences per 1,000 population','Fuel poverty','Excess winter deaths index','Violent crime - sexual offences per 1,000 population','Economic inactivity rate','Affordability of home ownership','Gender pay gap (by workplace location)','Average weekly earnings']] = minmax.fit_transform(nan_rows[['Year','popCount','Pupil absence','Violent crime - violence offences per 1,000 population','Fuel poverty','Excess winter deaths index','Violent crime - sexual offences per 1,000 population','Economic inactivity rate','Affordability of home ownership','Gender pay gap (by workplace location)','Average weekly earnings']])


# In[25]:


filtered_df.head()


# In[26]:


features = filtered_df.loc[:,filtered_df.columns != "obesity"].values


# In[27]:


labels = filtered_df['obesity']


# In[28]:


#X_test = nan_rows.loc[:,nan_rows.columns != "obesity"].values


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.1,random_state=101)


# In[30]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[31]:


y_pred = regressor.predict(X_test)


# In[32]:


y_pred[0:10]


# In[33]:


y_test[0:10]


# In[34]:


from sklearn.metrics import r2_score
r2_score_results = r2_score(y_test,y_pred)
print(r2_score_results)


# In[35]:


nan_rows['obesity']= regressor.predict(nan_rows.loc[:,nan_rows.columns != "obesity"].values)


# In[36]:


nan_rows = nan_rows.reset_index(drop=True)


# In[37]:


nan_rows.notnull().all()


# In[38]:


nan_rows.loc[:,nan_rows.columns != "obesity"].values.shape


# In[39]:


nan_rows


# In[40]:


nan_rows_copy['obesity'] = nan_rows['obesity']


# In[41]:


for row in range(len(nan_rows)):
    nan_rows_copy['obesity'][row] = nan_rows['obesity'][row]




# In[42]:


nan_rows_copy['obesity'] = nan_rows['obesity'].values


# In[43]:


nan_rows_copy['obesity'].round(1)


# In[44]:


new_df = filtered_df_copy.append(nan_rows_copy)


# In[45]:


new_df.round(1)


# In[46]:


new_df['Obesity in children per 100 children'] = new_df['obesity']/new_df['popCount']*100


# In[47]:


new_df


# In[48]:


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#import statsmodels.api as sm

#fig,axes = plt.subplots(nrows = 2,ncols = 1)

new_df.plot(y = 'Economic inactivity rate',x = 'Obesity in children per 100 children',kind = 'scatter',style = 'x',subplots = True,sharex = True,figsize = (20,20))
new_df.plot(y = 'Average weekly earnings',x = 'Obesity in children per 100 children',kind = 'scatter',style = 'x',subplots = True,sharex = True,figsize = (20,20))


# In[49]:


fig, ax = plt.subplots()
new_df.plot(x='Obesity in children per 100 children', y='Economic inactivity rate',style = 'x', ax=ax)


# In[50]:


# plot the data itself
from matplotlib import pylab
num_rows = 5
num_cols = 3
def Plot(y):

    x = new_df['Obesity in children per 100 children']
    #y = new_df['Economic inactivity rate']
    plt.figure(figsize=(2*3*num_cols, 2*num_rows))
    plt.plot(x,y,'x')

    # calc the trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    # the line equation:
    print ("y=%.6fx+(%.6f)"%(z[0],z[1]))
    plt.show()


# In[51]:


Plot(new_df['Fuel poverty'])


# In[52]:





# In[57]:


for i in range(len(new_df.columns.values.tolist())):
    i = i + 1
    print(i)
    Plot(new_df[new_df.columns.values[i]])
    plt.ylabel(new_df.columns.values[i])
    plt.xlabel("Obesity per 100 children")
    plt.show()
    if i == 13:
        break


# In[ ]:


new_df.columns.values[1]


# In[ ]:




