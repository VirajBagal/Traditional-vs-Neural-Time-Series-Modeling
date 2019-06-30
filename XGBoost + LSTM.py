# Preprocessing for XGBoost

low_date=data['Low']
low_date=pd.DataFrame(low_date,columns=['Low'])
low_date['Day']=low_date.index.day
low_date['Month']=low_date.index.month
low_date['Year']=low_date.index.year
low_date.reset_index(drop=True,inplace=True)

train_size=0.8
train_index=int(len(low_date)*train_size)
train=low_date.iloc[:train_index,:]
val=low_date.iloc[train_index:,:]
train_X=train.drop('Low',axis=1)
train_y=train['Low']
val_X=val.drop('Low',axis=1)
val_y=val['Low']


xgb_model=XGBRegressor(random_state=3)
xgb_model.fit(train_X,train_y)
pred=xgb_model.predict(val_X)
rmse=math.sqrt(mean_squared_error(val_y,pred))
val_size=val_X.shape[0]
print('RMSE for validation : '+ str(rmse/val_size)) 
#Normarlized with the length of validation set

train_pred=xgb_model.predict(train_X)
train_rmse=math.sqrt(mean_squared_error(train_y,train_pred))
train_size=train_X.shape[0]
print('RMSE for train : ' + str(train_rmse/train_size)) 
#Normalized with the length of train set

# We can see that the model is overfitting. You can use lower learning rate, tune number of estimators and use gamma for regularization to reduce overfitting. 
# I am using GridSearch to find best parameters. I am not sure whether this approach is correct or not because in GridSearch, we do crossvalidation and so, the order in the data is lost.
# Please provide your suggestions in the comments.

xgb1=XGBRegressor(random_state=5)
params={'n_estimators':np.arange(100,600,100),
       'learning_rate':np.arange(0.01,0.11,0.03),
       'gamma':np.arange(0,11,2),
       'subsample':[0.8]}

grid=GridSearchCV(xgb1,params,cv=5,scoring='neg_mean_squared_error',verbose=0)
grid.fit(train_X,train_y)

print(grid.best_params_)
print(grid.best_score_)

best_model=grid.best_estimator_
best_model.fit(train_X,train_y)
best_pred_train=best_model.predict(train_X)
train_rmse=math.sqrt(mean_squared_error(train_y,best_pred_train))
print('RMSE for training :' + str(train_rmse/train_size)) #Normalize with the length of train set
best_pred_val=best_model.predict(val_X)
val_rmse=math.sqrt(mean_squared_error(val_y,best_pred_val))
print('RMSE for validation :' + str(val_rmse/val_size)) #Normalize with the length of val set


#I guess it is still overfitting, but better than previous. You can try to optimize the parameters further.


#Deep Learning
#LSTM

#Data Preparation for LSTM

low=data['Low'].values
low=np.array(low).reshape(-1,1)

train_size=0.8
train_index=int(len(low)*train_size)
train=low[:train_index]
val=low[train_index:]

scl=MinMaxScaler(feature_range=(0,1))
train_scl=scl.fit_transform(train)
val_scl=scl.transform(val)

def prepare_dataset(data,window=30):
    X=[]
    Y=[]
    for i in range(len(data)-window):
        dummy_X=data[i:i+window]
        dummy_y=data[window+i]
        X.append(dummy_X)
        Y.append(dummy_y)
    return np.array(X),np.array(Y)


train_X,train_y=prepare_dataset(train_scl)
val_X,val_y=prepare_dataset(val_scl)

def rmse(true_y,pred_y):
    return backend.sqrt(backend.mean(backend.square(true_y-pred_y),axis=1))



window=30
input_layer=Input(shape=(window,1))
x=LSTM(4)(input_layer)
output=Dense(1,activation='linear')(x)
lstm_model=Model(input_layer,output)
lstm_model.compile(loss='mean_squared_error',optimizer='adam',metrics=[rmse])


result=lstm_model.fit(x=train_X,y=train_y,epochs=200,validation_data=[val_X,val_y])

lstm_train_rmse=result.history['rmse'][-1]
lstm_val_rmse=result.history['val_rmse'][-1]
print('RMSE for train :' + str(lstm_train_rmse/len(train_y)))  #Normalize with the length of train set
print('RMSE for validation :' + str(lstm_train_rmse/len(val_y)))  #Normalize with the length of val set

plt.subplots(2,1,figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(result.history['rmse'],label='training')
plt.plot(result.history['val_rmse'],label='validation',color='green')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(result.history['loss'],label='training')
plt.plot(result.history['val_loss'],label='validation',color='green')
plt.legend(loc='best')

# Things you can do further :

# 1. Optimize ARIMA parameters
# 2. Optimize XGBoost hyperparameters
# 3. Update the LSTM network.
# 4. Use different callbacks

