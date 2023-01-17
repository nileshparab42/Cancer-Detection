import pickle
import os 
import pandas as pd

load_model =pickle.load(open("SVC","rb"))

os.chdir ('C:\\Users\\Nilesh\Documents\\Cancer Detection Project')
data = pd.read_csv('test.csv')

x= data.drop (labels='Actual diagnosis' ,axis =1 )
x_norm = (x- x.mean()) / (x.max()- x.min())

load_model =pickle.load(open("SVC","rb"))
pred = load_model.predict(x_norm)

pred=pred.astype(str)

pred[pred=="1"]="Malignant"
pred[pred=="0"]="Benign"

data['Prediction'] = pred

data.to_csv('Result.csv')