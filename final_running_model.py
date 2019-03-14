import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
import pickle


data = pd.read_csv("C:/project/Manual-Data/Training.csv") 
data_alpha=pd.read_csv("C:/project/Manual-Data/Training.csv")
data_alpha=data_alpha.drop('prognosis', axis=1)
data_alpha= data_alpha.reindex(sorted(data_alpha.columns), axis=1)

x=data_alpha.iloc[:,:133]
y=data.iloc[:,132:]
num_classes=41


y1=np.ravel(y)
symptom=data_alpha.columns
disease=data['prognosis'].unique()


label_prognosis=LabelEncoder()
y_integer_encoded=label_prognosis.fit_transform(y1)
y_integer_encoded = y_integer_encoded.reshape(len(y_integer_encoded), 1)
y_new = keras.utils.to_categorical(y_integer_encoded, num_classes)


dummy_y = np_utils.to_categorical(y_integer_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=0.33, random_state=42)


""""model=Sequential()
model.add(Dense(input_dim=132,units=132,kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=153, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=153, kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train, batch_size=100, epochs=10)
predict=model.predict(x_test)"""




from numpy import argmax

from sklearn import preprocessing
lb = preprocessing.MultiLabelBinarizer()
lb.fit([symptom])
lb1=lb.transform([('watering_from_eyes','chills','shivering','continuous_sneezing',)])
#lb5=[('vomiting','watering_from_eyes','weakness_in_limbs','weakness_of_one_body_side','weight_gain','weight_loss','yellow_crust_ooze','yellow_urine','yellowing_of_eyes','yellowish_skin')]
lb4=np.array(lb1)
lb2=lb.transform([lb4])
print(lb2)

##lb1=lb.transform([('abdominal_pain','abnormal_menstruation')])

""""symptoms=lb.classes_
symptoms=np.array(symptoms)
predict_encoded=model.predict(lb2)
predict_encoded



inverted = label_prognosis.inverse_transform(argmax(predict_encoded[:,0:42]))

print(inverted)

percent=argmax(predict_encoded[:,0:42])
inverted_new = label_prognosis.inverse_transform(percent)


symptom


 with open('model_pickle','wb') as f:
	pickle.dump(model,f)"""

