import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
df=pd.read_csv('C:\\Users\\user\\Downloads\\spam_emails.csv')
df['Category']=df['Category'].map({'spam':1,'ham':0})
y=df['Category']
x=df['Message']
vector=TfidfVectorizer()
x_vector=vector.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_vector,y,test_size=0.20)
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_pred,y_test)
# Save model and vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vector, open("vector.pkl", "wb"))

print("Model and Vector saved successfully!")