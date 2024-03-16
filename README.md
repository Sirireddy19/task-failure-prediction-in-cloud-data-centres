# task-failure-prediction-in-cloud-data-centres

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
train_df=pd.read_csv('train_data.csv')
train_df.info()
print(train_df.shape)
import seaborn as sns
sns.countplot(x="failed", data = train_df)
Reshape X_train and X_test to 3D since Conv1D requires 3D data
X = train_df.iloc[:, 1:5]
y = train_df.iloc[:,5]
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train = X_train.values
X_test = X_test.values

X_train = X_train.reshape(-1, X_train.shape[1],1)
X_test = X_test.reshape(-1, X_test.shape[1],1)
print(X_train.shape)
print(X_test.shape)
Convert the Target label to categorical
target_train = y_train
target_test = y_test
Y_train=to_categorical(target_train)
Y_test=to_categorical(target_test)
print(Y_train.shape)
print(Y_test.shape)
Performance Evaluation Function
def showResults(test, pred):
    #target_names = ['positive', 'negative']
    # print(classification_report(test, pred, target_names=target_names))
    accuracy = accuracy_score(test, pred)
    precision=precision_score(test, pred, average='weighted')
    f1Score=f1_score(test, pred, average='weighted') 
    #loss=log_loss(test,pred)
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    #print("Loss : {}".format(loss))
    cm=confusion_matrix(test, pred)
    print(cm)
Model Defination
verbose, epoch, batch_size = 1, 100, 32
activationFunction='relu'

def getModel():
    
    cnnmodel = Sequential()
    #cnnmodel.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
    #cnnmodel.add(MaxPooling1D(pool_size=2))
    cnnmodel.add(Conv1D(filters=128, kernel_size=2, activation='relu',input_shape=(X_train.shape[1],X_train.shape[2])))
    cnnmodel.add(MaxPooling1D(pool_size=2))
    cnnmodel.add(Dropout(rate=0.2))
    cnnmodel.add(Flatten())
    cnnmodel.add(Dense(64, activation='relu'))
    cnnmodel.add(Dense(32, activation='relu'))
    cnnmodel.add(Dense(2, activation='softmax'))
    cnnmodel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    cnnmodel.summary()
    return cnnmodel

cnnmodel = getModel()
modelhistory= cnnmodel.fit(X_train, Y_train, epochs=20,steps_per_epoch=2, verbose=verbose, validation_split=0.2, batch_size = batch_size)
Plot the Training Accuracy and Loss
# Plot training & validation loss values
plt.plot(modelhistory.history['loss'])
plt.plot(modelhistory.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(modelhistory.history['accuracy'])
plt.plot(modelhistory.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()
Test the Model
cnnpredictions = cnnmodel.predict(X_test, verbose=1)
Evaluate the Performance
cnn_predict=np.argmax(cnnpredictions,axis=1)
cnn_actual_value=np.argmax(Y_test,axis=1)
showResults(cnn_actual_value, cnn_predict)
cnn = accuracy_score(cnn_actual_value, cnn_predict)
f1cnn=f1_score(cnn_actual_value, cnn_predict, average='weighted') 
# KFold VAlidation
tempX=train_df.iloc[:,1:5].values
tempY=train_df.iloc[:,5].values
skf = StratifiedKFold(n_splits=3, shuffle=True)
skf.get_n_splits(X_train, target_train)
foldNum=0
for train_index, val_index in skf.split(X_train, target_train):
    foldNum+=1
    print("Results for fold",foldNum)
    X_traincv, X_valcv = tempX[train_index], tempX[val_index]
    Y_traincv, Y_valcv = tempY[train_index], tempY[val_index]

    X_traincv=X_traincv.reshape(X_traincv.shape[0],X_traincv.shape[1],1)
    X_valcv=X_valcv.reshape(X_valcv.shape[0],X_valcv.shape[1],1)
    # one hot encode
    Y_traincv = to_categorical(Y_traincv)
    Y_valcv = to_categorical(Y_valcv)
    
    history = cnnmodel.fit(X_traincv, Y_traincv, validation_data = (X_valcv, Y_valcv), epochs=20,steps_per_epoch=2, batch_size=batch_size)  
    yPredict = cnnmodel.predict(X_valcv)

    #Converting one hot encoded test label to label    
    pred = np.argmax(yPredict, axis=1)
    val = np.argmax(Y_valcv, axis=1)
    
    showResults(val, pred)
Test the Model
cnnpredictionscv = cnnmodel.predict(X_test, verbose=1)
Evaluate the Performance
cnn_predictcv=np.argmax(cnnpredictionscv,axis=1)
cnn_actual_valuecv=np.argmax(Y_test,axis=1)
showResults(cnn_actual_valuecv, cnn_predictcv)
cnnkv = accuracy_score(cnn_actual_valuecv, cnn_predictcv)
f1cnnkv=f1_score(cnn_actual_valuecv, cnn_predictcv, average='weighted') 
# **Hybrid CNN LSTM**
import tensorflow as tf
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding="causal",activation="relu",input_shape=(X_train.shape[1],X_train.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="valid"),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2)
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(5e-4,
                                                             decay_steps=1000000,
                                                             decay_rate=0.98,
                                                             staircase=False)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
              metrics=['acc'])
model.summary()
history = model.fit(X_train, Y_train,epochs=10)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()
predictions = model.predict(X_test, verbose=1)
predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)
hyd = accuracy_score(actual_valuecv, predictcv)
f1hyd=f1_score(actual_valuecv, predictcv, average='weighted') 
# LSTM
from keras.callbacks import EarlyStopping
import math
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
es=EarlyStopping(patience=7)
model=Sequential()
model.add(LSTM(13,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(units=2))
model.add(Activation('sigmoid'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
              metrics=['acc'])
model.summary()
history = model.fit(X_train, Y_train,epochs=10)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()
predictions = model.predict(X_test, verbose=1)
predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)
lstm = accuracy_score(actual_valuecv, predictcv)
f1lstm=f1_score(actual_valuecv, predictcv, average='weighted') 
# BiLSTM
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
es=EarlyStopping(patience=7)
model=Sequential()
model.add(Bidirectional(LSTM(13,input_shape=(X_train.shape[1],X_train.shape[2]))))
model.add(Dense(units=2))
model.add(Activation('sigmoid'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
              metrics=['acc'])

history = model.fit(X_train, Y_train,epochs=10)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()
predictions = model.predict(X_test, verbose=1)
predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)
bilstm = accuracy_score(actual_valuecv, predictcv)
f1bilstm=f1_score(actual_valuecv, predictcv, average='weighted') 
# RNN
from keras.layers import Dense, Dropout, Embedding, SimpleRNN
es=EarlyStopping(patience=7)
model=Sequential()
model.add(SimpleRNN(13,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(units=2))
model.add(Activation('sigmoid'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.8),
              metrics=['acc'])

model.summary()
history = model.fit(X_train, Y_train,epochs=10)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('loss.png', format='png', dpi=1200)
plt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('accuracy.png', format='png', dpi=1200)
plt.show()
predictions = model.predict(X_test, verbose=1)
predictcv=np.argmax(predictions,axis=1)
actual_valuecv=np.argmax(Y_test,axis=1)
showResults(actual_valuecv, predictcv)
rnn = accuracy_score(actual_valuecv, predictcv)
f1rnn=f1_score(actual_valuecv, predictcv, average='weighted') 
# For Machine Learning
X = train_df.iloc[:, 1:5]
y = train_df.iloc[:,5]
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X, y)
predictions = RF.predict(X)
print("*Confusion Matrix for RF: ")
print(confusion_matrix(y, predictions))
print("*Classification report for RF: ")
print(classification_report(y, predictions))
val1 = accuracy_score(y, predictions) *100
print(val1)
f1rf = f1_score(y, predictions, average='weighted') 
# Decision Tree
from sklearn import tree
DT = tree.DecisionTreeClassifier()
DT.fit(X, y)
predictions = DT.predict(X)
print("*Confusion Matrix for DT: ")
print(confusion_matrix(y, predictions))
print("*Classification report for DT: ")
print(classification_report(y, predictions))
val2 = accuracy_score(y, predictions) *100
print(val2)
f1dt = f1_score(y, predictions, average='weighted') 
# Hidden Markov Model
!pip install hmmlearn
from hmmlearn.hmm import GaussianHMM
hmm = GaussianHMM(n_components=2)
hmm.fit(X)
predictions = hmm.predict(X)
print("*Confusion Matrix for HMM: ")
print(confusion_matrix(y, predictions))
print("*Classification report for HMM: ")
print(classification_report(y, predictions))
val3 = accuracy_score(y, predictions) *100
print(val3)
f1hmm = f1_score(y, predictions, average='weighted') 
# Support Vector Machine
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X, y)
predictions = clf.predict(X)
print("*Confusion Matrix for SVM: ")
print(confusion_matrix(y, predictions))

val4 = accuracy_score(y, predictions) *100
print(val4)
print("*Classification report for SVM: ")
print(classification_report(y, predictions))
f1svm = f1_score(y, predictions, average='weighted') 
# Voting Classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = SVC(gamma='auto')
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = DecisionTreeClassifier()
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')
eclf1.fit(X, y)
predictions = eclf1.predict(X)
print("*Confusion Matrix for Voting Classifier: ")
print(confusion_matrix(y, predictions))
print("*Classification report for Voting Classifier: ")
print(classification_report(y, predictions))
val5 = accuracy_score(y, predictions) *100
print(val5)
f1vot = f1_score(y, predictions, average='weighted') 
# Comparison
import numpy as np
import matplotlib.pyplot as plt

N = 11
ind = np.arange(N)  # the x locations for the groups
width = 0.3      # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [cnn*100,cnnkv*100,hyd*100,lstm*100,bilstm*100,rnn*100,val1,val2,val3,val4,val5]

rects1 = ax.bar(ind, yvals, width, color='r')
zvals = [f1cnn*100,f1cnnkv*100,f1hyd*100,f1lstm*100,f1bilstm*100,f1rnn*100,f1rf*100,f1dt*100,f1hmm*100,f1svm*100,f1vot*100]
rects2 = ax.bar(ind+width, zvals, width, color='g')


ax.set_ylabel('Scores')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('CNN','CNN+KFlodValidation','CNN+LSTM','LSTM','BiLSTM','RNN','RF','DT','HMM','SVM','Voting Classifier') )
ax.legend( (rects1[0], rects2[0]), ('Accuracy', 'F1Score') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
score = [cnn*100,cnnkv*100,hyd*100,lstm*100,bilstm*100,rnn*100,val1,val2,val3,val4,val5]
#make variabel for save the result and to show it
classifier = ('CNN','CNN+KFlodValidation','CNN+LSTM','LSTM','Voting classifier','RNN','RF','DT','HMM','SVM','BiLSTM')
y_pos = np.arange(len(classifier))
print(y_pos)
print(score)
import matplotlib.pyplot as plt2
plt2.barh(y_pos, score, align='center', alpha=0.5,color='blue')
plt2.yticks(y_pos, classifier)
plt2.xlabel('Score')
plt2.title('Classification Performance')
plt2.show()
import joblib
filename = 'model.sav'
joblib.dump(eclf1, filename)
