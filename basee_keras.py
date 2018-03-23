#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

sns.set_style("whitegrid")
np.random.seed(697)

df = pd.read_csv('./input/train_sample.csv', header = 0)

print(df.is_attributed.value_counts()) #very imbalanced data set

#---------------------------Pre-processing-------------------------
#Create new variables
df['ip_cut'] = pd.cut(df.ip,15)
df['time_interval'] = df.click_time.str[11:13]

#Drop unneeded variables
df = df.drop(['ip', 'attributed_time', 'click_time'], axis = 1)


#Encode categorical variables to ONE-HOT
categorical_columns = ['app', 'device', 'os', 'channel', 'ip_cut', 'time_interval']

df = pd.get_dummies(df, columns = categorical_columns)     

#Split in 75% train and 25% test set
train_df, test_df = train_test_split(df, test_size = 0.25, random_state= 1984)

#Make sure labels are equally distributed in train and test set
train_df.is_attributed.sum()/train_df.shape[0] #0.2233
test_df.is_attributed.sum()/test_df.shape[0] #0.2148

#Get the data ready for the Neural Network
train_y = train_df.is_attributed
test_y = test_df.is_attributed

train_x = train_df.drop(['is_attributed'], axis = 1)
test_x = test_df.drop(['is_attributed'], axis = 1)

train_x =np.array(train_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
test_y = np.array(test_y)

#-------------------Build the Neural Network model-------------------
print('Building Neural Network model...')
adam = optimizers.adam(lr = 0.005, decay = 0.0000001)

model = Sequential()
model.add(Dense(48, input_dim=train_x.shape[1],
                kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(24,
                #kernel_regularizer=regularizers.l2(0.02),
                activation="tanh"))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(train_x, train_y, validation_split=0.1, epochs=3, batch_size=64)


#Predict on test set
predictions_NN_prob = model.predict(test_x)
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)