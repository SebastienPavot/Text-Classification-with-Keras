#Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer


st.set_option('deprecation.showPyplotGlobalUse', False)

#Data importation
train = pd.read_csv('/Users/spavot/Documents/Perso/Text classification & Visualization/Data/Corona_NLP_train.csv', encoding = 'latin')
test = pd.read_csv('/Users/spavot/Documents/Perso/Text classification & Visualization/Data/Corona_NLP_test.csv', encoding ='latin')

#Title
st.title('Text classification with Keras')

#Overview of the dataset:
st.write("""
The original data: Covid-19 tweets related:
""")

#Shape and head:
st.write('Shape of train dataset:', train.shape)
st.write('Shape of test set:', test.shape)
st.write('Quick overview of the data:', train.head())

#Select the model=
selected_NN = st.sidebar.selectbox("Select the Neural Network", ('Simple NN', 'Multi layers NN', 'Embedded Multi layers NN', 'Embbed Max pool Multi Layers NN', 
                            'CNN Multi layers model'))

train_text = train.OriginalTweet.values
test_text = test.OriginalTweet.values

vectorizer = CountVectorizer()
vectorizer.fit(train_text)

#Create training / test set:
X_train = vectorizer.transform(train_text)
X_test  = vectorizer.transform(test_text)
y_train = pd.get_dummies(train.Sentiment).values
y_test = pd.get_dummies(test.Sentiment).values

model_simple = Sequential()
model_simple.add(Dense(30, input_dim = X_train.shape[1], activation = 'relu'))
model_simple.add(Dense(5, activation = 'softmax'))
model_simple.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

st.sidebar.write("Summary of the model:", model_simple.summary())
