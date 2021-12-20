import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("deploy_df.csv")


df.drop(['Flight_Date','FlyTime'],axis=1,inplace=True)

x=df.drop('Drone_Cost',axis=1)
y=df['Drone_Cost']
##################MODEL BUILDING#################################

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

cat=CatBoostRegressor()
cat.fit(X_train,y_train)
cat_pred=cat.predict(X_test)
print("CATBOOST PREDICTION SCORE:-->",r2_score(y_test,cat_pred)*100)

import pickle
#Saving the model to our drive
pickle.dump(cat,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))














#cat=CatBoostRegressor()
#cat.fit(X_train,y_train)

