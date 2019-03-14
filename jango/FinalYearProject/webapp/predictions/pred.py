import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
import pickle
from numpy import argmax
from sklearn import preprocessing
from sklearn.externals import joblib



def disease_prediction(sym1,sym2,sym3,sym4,sym5,sym6,sym7,sym8):

	symptom1=sym1
	symptom2=sym2
	symptom3=sym3
	symptom4=sym4
	symptom5=sym5
	symptom6=sym6
	symptom7=sym7
	symptom8=sym8 

	check = 'none'
	data = pd.read_csv("C:/project/Manual-Data/Training.csv")

	data_alpha = pd.read_csv("C:/project/Manual-Data/Training.csv")

	data_alpha = data_alpha.drop('prognosis', axis = 1)

	data_alpha.reindex(sorted(data_alpha.columns), axis = 1)


	x = data_alpha.iloc[:,:133]

	y = data.iloc[:,132:]

	num_classes = 41

	y1 = np.ravel(y)

	symptom = data_alpha.columns 

	disease = data['prognosis'].unique

	label_prognosis = LabelEncoder()

	y_integer_encoded = label_prognosis.fit_transform(y1)

	y_integer_encoded = y_integer_encoded.reshape(len(y_integer_encoded), 1)

	y_new = keras.utils.to_categorical(y_integer_encoded, num_classes)

	symptoms = []

	if(symptom1!=check):
		symptoms.append(symptom1)

	if(symptom2!=check):
		symptoms.append(symptom2)

	if(symptom3!=check):
		symptoms.append(symptom3)

	if(symptom4!=check):
		symptoms.append(symptom4)

	if(symptom5!=check):
		symptoms.append(symptom5)

	if(symptom6!=check):
		symptoms.append(symptom6)

	if(symptom7!=check):
		symptoms.append(symptom7)

	if(symptom8!=check):
		symptoms.append(symptom8)

	symptoms = np.array(symptoms)	

	lb = preprocessing.MultiLabelBinarizer()

	lb.fit([symptom])

	lb1 = lb.transform([symptoms])

	with open('C:/project/jango/FinalYearProject/webapp/model_pickle','rb') as f:
		model_prediction = pickle.load(f)

	predict_encoded = model_prediction.predict(lb1)
	
	inverted = label_prognosis.inverse_transform(argmax(predict_encoded[:,0:40]))

	return inverted	







	   	





	



