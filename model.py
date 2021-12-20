import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
df=pd.read_excel("Data_Train.xlsx")


df.dropna(how='any',inplace=True)
df['Date_of_Journey']=pd.to_datetime(df['Date_of_Journey'])
df['Day_of_Journey']=(df['Date_of_Journey']).dt.day
df['Month_of_Journey']=(df['Date_of_Journey']).dt.month
df['Dep_hr']=pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min']=pd.to_datetime(df['Dep_Time']).dt.minute
df.drop(['Dep_Time'],axis=1,inplace=True)
df['Arrival_hr']=pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min']=pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop(['Arrival_Time'],axis=1,inplace=True)

duration=df['Duration'].str.split(' ',expand=True) #Spliting the duration from the datapoint
duration[1].fillna('00m',inplace=True) #Filling the nan values with 00m
df['duration_hr']=duration[0].apply(lambda x:x[:-1])  #Will skip h and only take hour part
df['duration_min']=duration[1].apply(lambda x:x[:-1]) #will skip m and only take the minute part

df.groupby(['Airline','Total_Stops'])['Price'].mean()
for i in df:
    df.replace('New Delhi','Delhi',inplace=True)

df['Total_Stops'].unique()
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0,'2 stops':2,'1 stop':1,'3 stops':3,'4 stops':4})
air_dummy=pd.get_dummies(df['Airline'],drop_first=True)

source_dest_dummy=pd.get_dummies(df[['Source','Destination']],drop_first=True)
df=pd.concat([air_dummy,source_dest_dummy,df],axis=1)

df.drop(['Airline','Source','Destination'],inplace=True,axis=1)


df_test=pd.read_excel('Test_set.xlsx')

df_test['Date_of_Journey']=pd.to_datetime(df_test['Date_of_Journey'])
df_test['Day_of_Journey']=(df_test['Date_of_Journey']).dt.day
df_test['Month_of_Journey']=(df_test['Date_of_Journey']).dt.month
df_test.drop(['Date_of_Journey'],axis=1,inplace=True)
#DepTIme
df_test['Dep_hr']=pd.to_datetime(df_test['Dep_Time']).dt.hour
df_test['Dep_min']=pd.to_datetime(df_test['Dep_Time']).dt.minute
df_test.drop(['Dep_Time'],axis=1,inplace=True)

#Arrival Time
df_test['Arrival_hr']=pd.to_datetime(df_test['Arrival_Time']).dt.hour
df_test['Arrival_min']=pd.to_datetime(df_test['Arrival_Time']).dt.minute
df_test.drop(['Arrival_Time'],axis=1,inplace=True)
#Splitting Time
duration=df_test['Duration'].str.split(' ',expand=True) #Spliting the duration from the datapoint
duration[1].fillna('00m',inplace=True) #Filling the nan values with 00m
df_test['duration_hr']=duration[0].apply(lambda x:x[:-1])  #Will skip h and only take hour part
df_test['duration_min']=duration[1].apply(lambda x:x[:-1]) #will skip m and only take the minute part
df_test['Airline'].value_counts()

#df_test.groupby(['Airline','Total_Stops'])['Price'].mean()

for i in df_test:
    df_test.replace('New Delhi','Delhi',inplace=True)

df_test['Destination'].unique()

#df_test['Total_Stops'].unique()
df_test['Total_Stops']=df_test['Total_Stops'].map({'non-stop':0,'2 stops':2,'1 stop':1,'3 stops':3,'4 stops':4})

air_dummy=pd.get_dummies(df_test['Airline'],drop_first=True)
source_dest_dummy=pd.get_dummies(df_test[['Source','Destination']],drop_first=True)
df_test=pd.concat([air_dummy,source_dest_dummy,df_test],axis=1)

df_test.drop(['Source','Destination','Additional_Info','Route'],axis=1,inplace=True)
df_test.drop(['Airline'],axis=1,inplace=True)


x=df.drop(['Route','Price','Additional_Info','Date_of_Journey','Duration'],axis=1)
y=df['Price']


##################MODEL BUILDING#################################

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

cat=CatBoostRegressor()
cat.fit(X_train,y_train)
cat_pred=cat.predict(X_test)
print("CATBOOST PREDICTION SCORE:-->",r2_score(y_test,cat_pred))

df=pd.read_csv('deploy_df.csv')

x=df.drop('Price',axis=1)
y=df['Price']

import pickle
#Saving the model to our drive
pickle.dump(cat,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))














#cat=CatBoostRegressor()
#cat.fit(X_train,y_train)

