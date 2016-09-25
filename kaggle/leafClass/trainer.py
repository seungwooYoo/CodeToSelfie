import pandas as pd
import os
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from datetime import date 
import ipdb

data = pd.read_csv('./data/train.csv')
parent_data = data.copy()

data_id = data.pop('id')
y = data.pop('species')
y = LabelEncoder().fit(y).transform(y)
print 'y.shape : {}'.format(y.shape)

X = StandardScaler().fit(data).transform(data)
print 'X.shape : {}'.format(X.shape)

y_cat = to_categorical(y)
print 'y.categorical : {}'.format(y_cat.shape)

"""
    Developing NN
"""
model = Sequential()
model.add(Dense(1024, input_dim=192))
model.add(Dropout(0.1))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(99))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
history = model.fit(X, y_cat, batch_size=129, nb_epoch=100, verbose=0)

plt.plot(history.history['loss'], 'o-')
plt.xlabel('Number of Iterations')
plt.ylabel('Categorical Crossentropy')
plt.title('Train error vs number of iterations')
plt.savefig('nn_history.png')

test = pd.read_csv('./data/test.csv')
index = test.pop('id')
test = StandardScaler().fit(test).transform(test)

yPred = model.predict_proba(test)
#ipdb.set_trace()
columns = parent_data.species.unique().tolist()
columns.sort()
yPred = pd.DataFrame(yPred, index=index, columns=columns)

cur_date = date.today()
year = cur_date.year
day = cur_date.day
month = cur_date.month

submit_file_name = 'submission_' + str(day) + str(month) + str(year) + '.csv'
fp = open(submit_file_name, 'wt')
fp.write(yPred.to_csv())
