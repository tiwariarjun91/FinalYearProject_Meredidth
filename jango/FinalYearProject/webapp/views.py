from django.shortcuts import render
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
from webapp.predictions.pred import *

# Create your views here.
def login(request):
	return render(request, 'login.html')


def submit(request):
	sym1 = request.POST['symptom1']
	sym2 = request.POST['symptom2']
	sym3 = request.POST['symptom3']
	sym4 = request.POST['symptom4']
	sym5 = request.POST['symptom5']
	sym6 = request.POST['symptom6']
	sym7 = request.POST['symptom7']
	sym8 = request.POST['symptom8']
	num3 = disease_prediction(sym1,sym2,sym3,sym4,sym5,sym6,sym7,sym8)
	
	return render(request, 'submit.html',{'num3':num3})
