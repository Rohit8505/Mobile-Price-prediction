#!/usr/bin/env python
# coding: utf-8

# # Mobile data

# Analyzing the mobile data sheet of 2023 provides valuable insights into the influence of mobile brand prices, brand popularity, price categories, sales figures, and market dominance. By delving into this information, we can ascertain which brand commands greater popularity and establish which price category they belong to. Furthermore, we can determine the brand with the highest sales and the strongest market foothold.
# 
# By examining the data, we can discern the correlation between mobile brand prices and their respective popularity, allowing us to gauge the preferences of consumers. Additionally, categorizing the price ranges enables us to understand the market positioning of each brand, providing a comprehensive overview of their pricing strategies.
# 
# Furthermore, the data reveals the brand that stands out with the highest sales figures, indicative of their strong customer base and market presence. This brand's ability to capture the attention and trust of consumers showcases their superior marketing strategies and product offerings, solidifying their position as a leader in the mobile industry.
# 
# In conclusion, analyzing the 2023 mobile data sheet uncovers essential information about the impact of brand prices, brand popularity, price categories, sales performance, and market dominance. This comprehensive analysis empowers us to make informed judgments about which brand is more popular, which price category they belong to, and which brand boasts the highest sales and strongest market hold.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


mob_data = pd.read_csv('mobile_prices_2023.csv')
print(mob_data.shape)


# In[3]:


mob_data.head(10)


# We will split the "Phone Name" column in our dataset, which is in the format (Brand Model ModelColorRom), and create new columns for the brand name, model, and color.

# In[4]:


mob_data.tail()


# In[5]:


mob_data.dtypes


# In[6]:


split_brand = mob_data['Phone Name'].str.split(' ', n=1, expand = True)
mob_data['Brand'] = split_brand[0]
mob_data.head()


# In[7]:


split_brand = mob_data['Phone Name'].str.split(' ', n=1, expand = True)
mob_data['Details']  = split_brand[1]


# In[8]:


split_mod = mob_data['Details'].str.split('(', n=1, expand=True)
mob_data['Model'] = split_mod[0]
mob_data


# In[9]:


split_mod = mob_data['Details'].str.split('(', n=1, expand=True)
mob_data['Extra']= split_mod[1]


# In[10]:


Split_color = mob_data['Extra'].str.split(",",n=1,expand=True)
mob_data['Color'] = Split_color[0]
mob_data.head()


# In[11]:


mob_data.columns


# In[12]:


mob_data.drop(['Extra','Details','Date of Scraping','Phone Name'],inplace=True, axis=1)


# In[13]:


mob_data.columns


# In[14]:


mob_data.rename(columns={'Rating ?/5':'Rating','Number of Ratings':'Total_Rating','ROM/Storage':'ROM','Back/Rare Camera':"Rare_camera",'Price in INR':'Price_INR','Battery':'Battery_mAH'},inplace=True)


# In[15]:


mob_data.head()


# From Phone Name col we ceated three col names brand, model no and color col for our further analysis which brand has the highest sales etc.

# In[16]:


#checking data types of the datasets features
mob_data.dtypes


# In[17]:


mob_data['Total_Rating']=mob_data['Total_Rating'].str.replace(',','').astype('int')


# In[18]:


mob_data.head()


# In[19]:


def currency_numeric (value):
    value = str(value)
    value=value.replace('₹', '').replace(',','')
    try:
        return int(value)
    except ValueError:
        return None


# In[20]:


mob_data["Price_INR"] = mob_data["Price_INR"].apply(currency_numeric)


# In[21]:


mob_data.dtypes


# In[22]:


import missingno as msno
mob_data.describe()


# In[23]:


mob_data.info()


# In[24]:


mob_data.isnull().sum()


# In[25]:


mob_data.dropna(inplace=True)


# In[26]:


mob_data.isnull().sum()


# In[27]:


msno.matrix(mob_data)
plt.show()


# In[28]:


def mah (value):
    value = str(value)
    value=value.replace('mAh', '')
    try:
        return int(value)
    except ValueError:
        return None


# In[29]:


mob_data["Battery_mAH"] = mob_data["Battery_mAH"].apply(mah)


# In[30]:


mob_data.Brand.unique()


# In[31]:


mob_data.replace('realme','Realme',inplace=True)
mob_data.replace('10A','Realme',inplace=True)
mob_data.replace('Moto','MOTOROLA',inplace=True)


# In[32]:


mob_data.head()    


# In[ ]:





# In[ ]:





# In[33]:


#visualization Of the movile data set


# In[34]:


#Brand_vs_Rating
import seaborn as sns
plt.style.use('ggplot')


# In[35]:


Brand_avg_price = mob_data.groupby('Brand')['Price_INR'].mean()


# In[36]:


Brand_avg_price = Brand_avg_price.sort_values(ascending=False)
Brand_avg_price.plot(kind='bar')
plt.title('Average Price')
plt.xlabel('Brand Name')
plt.ylabel('Average Price')
plt.show()


# In[37]:


# As per price range no of model of mobile phones
Brand_no_of_model = mob_data.groupby('Brand').agg(No_of_model=('Model', 'count'))
brand_no_of_model = Brand_no_of_model.sort_values(by= 'No_of_model',ascending = False)


# In[38]:


brand_no_of_model.plot(kind='barh')
plt.title('Brand Vs No of models')
plt.xlabel('No. of Models')
plt.ylabel('Brands')
plt.show()


# In[39]:


sorted_data = mob_data.sort_values(by='Rating',ascending=False).reset_index()
brand_model_sorted = sorted_data[['Brand', 'Model', 'Rating']].drop_duplicates()


# In[40]:


brand_model_sorted.head(20)


# In[41]:


sorted_data = mob_data.sort_values(by='Total_Rating',ascending=False).reset_index()
brand_model_max_rating = sorted_data[['Brand', 'Model', 'Rating','Total_Rating']].drop_duplicates()
brand_model_max_rating.head(20)


# In[ ]:





# In[ ]:





# In[42]:


## prediction for which price is effected


# In[43]:


df = mob_data.copy()


# In[44]:


df.head()


# In[45]:


df.info()


# In[46]:


df.describe()


# In[ ]:





# In[47]:


df.columns


# In[48]:


df2 = df[['RAM', 'ROM', 'Rare_camera', 'Front Camera',
       'Battery_mAH', 'Processor', 'Price_INR']]


# In[49]:


df2.head()


# In[50]:


split_ram = df2['RAM'].str.split(' ',n=1)
df2['RAM'] =df2['RAM'].str.split(' ',n=1).str[0].astype(float)


# In[51]:


df2['ROM'] =df2['ROM'].str.split(' ',n=1).str[0].astype(int)


# In[52]:


df2.dtypes


# In[53]:


df2.head()


# In[54]:


df2 = pd.get_dummies(df2, columns=['Rare_camera'], prefix ='rare', drop_first=True)
df2 = pd.get_dummies(df2, columns=['Front Camera'], prefix ='Front Camera', drop_first=True)
df2=  pd.get_dummies(df2, columns=['Processor'], prefix ='Processor', drop_first=True)


# In[59]:


df2.columns


# In[56]:


df2.shape


# In[60]:


X = df2.drop('Price_INR',axis=1)
y = df2['Price_INR']


# In[62]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[63]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[64]:


from sklearn.svm import SVC

model = SVC()
model.fit(X_train_scaled, y_train)


# In[65]:


y_pred = model.predict(X_test_scaled)


# In[66]:


y_pred = model.predict(X_test_scaled)


# In[68]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In conclusion, the tested model achieved an accuracy of 24% in predicting mobile data. This indicates that factors beyond those considered in the model, such as body metal, OS, and others, significantly influence mobile pricing. A more comprehensive approach with a broader range of features is needed to improve predictive accuracy and gain deeper insights into consumer preferences and pricing trends in the mobile industry.

# In[ ]:




