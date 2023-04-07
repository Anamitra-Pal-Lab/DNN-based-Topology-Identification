from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import regularizers
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import time  
from sklearn.metrics import r2_score
from tensorflow.keras import losses
import random as python_random
import tensorflow as tf
from sklearn.linear_model import LinearRegression
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)


dfx_train = pd.read_csv('temp1.csv')
dfy_train = pd.read_csv('temp2.csv')
x_train = dfx_train.to_numpy()
y_train = dfy_train.to_numpy()
# y_train_open = np.array([0]*x_train_switch_open.shape[0])
# y_train_closed = np.array([1]*x_train_switch_closed.shape[0])
# x_train_all_temp = np.concatenate((x_train_switch_open,x_train_switch_closed),0)
# y_train_all_temp = np.concatenate((y_train_open,y_train_closed),0)  
# y_train_all_temp = np.reshape(y_train_all_temp, (y_train_all_temp.shape[0], 1))  
# x_train_switch , y_train_switch = shuffle(x_train_all_temp, y_train_all_temp)

dfx_test = pd.read_csv('temp3.csv')
dfy_test = pd.read_csv('temp4.csv')
x_test = dfx_test.to_numpy()
y_test = dfy_test.to_numpy()

y_train = y_train[:,0] # second parameter should be zero for categorical and 1:10 for binary
y_test = y_test[:,0] # second parameter should be zero for categorical and 1:10 for binary
y_train = y_train -1
y_test = y_test -1
# y_test_open = np.array([0]*x_test_switch_open.shape[0])
# y_test_closed = np.array([1]*x_test_switch_closed.shape[0])
# x_test_all_temp = np.concatenate((x_test_switch_open,x_test_switch_closed),0)
# y_test_all_temp = np.concatenate((y_test_open,y_test_closed),0)
# y_test_all_temp = np.reshape(y_test_all_temp, (y_test_all_temp.shape[0], 1)) 
# x_test_switch , y_test_switch = shuffle(x_test_all_temp, y_test_all_temp)

#### determining PMU location######################################
x_train = x_train[:,6:]
x_test = x_test[:,6:]
# temp1 = np.arange(49,49+6)
# temp2 = np.arange(73,73+6)
# temp3 = np.arange(85,85+6)
# temp4 = np.arange(91,91+6)
#temp5 = np.arange(49,49+3)
#temp4 = np.arange(34,34+3)
#temp4 = np.arange(19,19+3)
#temp5 = temp3
#temp5 = np.concatenate((temp3,temp4), axis=0)
# temp5 = np.concatenate((temp1,temp3,temp4), axis=0)
#temp5 = np.concatenate((temp1,temp2,temp3,temp4), axis=0) 
#temp5 = np.concatenate((temp1,temp2,temp3,temp4,temp5), axis=0)
#temp5 = np.reshape(temp5,(1,15)) # only 1 mPMU and only voltage
#temp5 = np.reshape(temp5,(1,9)) # only 1 mPMU
# temp5 = np.reshape(temp5,(1,18))
# temp5 = temp5.astype(int)
# temp5 = temp5-1
# temp5 = temp5[0]
# x_train = x_train[:,temp5]
# x_test = x_test[:,temp5]
# temp5 = np.array([5,10,14,16,19])-1
#x_train = x_train[:,6:]
#x_test = x_test[:,6:]



#Build the neural network
model_switch = Sequential()
##model.add(Dense(400, input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.01) ,activation='relu')) # Hidden 1
model_switch.add(Dense(800, input_dim=x_train.shape[1] ,activation='relu')) # Hidden 1
model_switch.add(BatchNormalization())
model_switch.add(Dropout(0.3))
model_switch.add(Dense(800, activation='relu')) # Hidden 2
model_switch.add(BatchNormalization())
model_switch.add(Dropout(0.3))
model_switch.add(Dense(800, activation='relu')) # Hidden 3
model_switch.add(BatchNormalization())
model_switch.add(Dropout(0.3))
model_switch.add(Dense(800, activation='relu')) # Hidden 4
model_switch.add(BatchNormalization())
model_switch.add(Dropout(0.3))
model_switch.add(Dense(800, activation='relu')) # Hidden 5
model_switch.add(BatchNormalization())
model_switch.add(Dropout(0.3))
model_switch.add(Dense(85, activation='softmax')) # Output
#
  
y_binary = to_categorical(y_train) 
#for j in np.linspace (0.1,0.3,1):
Adam(learning_rate=0.02726, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_switch.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ES = EarlyStopping (monitor='val_loss',patience=50,verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001)
filepath="TI_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
#history_switch = model_switch.fit(x_train,y_train,verbose=1,epochs=5,validation_split=0.2,callbacks=[ES,reduce_lr])
history_switch = model_switch.fit(x_train,y_binary,verbose=1,batch_size=105,epochs=50,validation_split=0.2, callbacks=[checkpoint,reduce_lr])
y_test_binary = to_categorical( y_test) 
model_switch.load_weights("TI_weights.best.hdf5")
pred_switch = model_switch.predict(x_test)
pred_switch = np.argmax(pred_switch,axis=1)
y_compare = np.argmax(y_test_binary,axis=1) 
score = metrics.accuracy_score(y_compare, pred_switch)


