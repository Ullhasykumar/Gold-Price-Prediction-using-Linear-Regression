# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

gold_data=yf.download("GLD",start='2020-01-01',end='2026-06-30')

gold=pd.DataFrame(gold_data)
gold.head()

X,y=gold[['High','Low','Open']],gold['Close']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
y_train=np.array(y_train)
y_test=np.array(y_test)

class myLR:
  def __init__(self):
    self.coef_ = None
    self.intercept_ = None

  def fit(self, X_train, y_train):
    X_train=np.insert(X_train,0,1,axis=1)

    betas=np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
    print(betas)

    self.intercept_=betas[0]
    self.coef_=betas[1:]

  def predict(self,X_test):
    y_pred=np.dot(X_test,self.coef_)+self.intercept_
    return y_pred

lr=myLR()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

comparsion=pd.DataFrame(np.c_[y_test,y_pred],columns=['Actual','Predicted'])
comparsion
