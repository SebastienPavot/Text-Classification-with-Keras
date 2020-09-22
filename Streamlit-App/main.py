#Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import plotly.graph_objects as go


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
st.write('Quick overview of the data:')
st.table(train.head())

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
model_simple.add(Dense(30, input_dim = X_train.shape[1], activation = 'relu', name = 'Input_Dense_30'))
model_simple.add(Dense(30, activation = 'relu', name = 'Dense_30_First'))
model_simple.add(Dense(30, activation = 'relu', name = 'Dense_30_Second'))
model_simple.add(Dense(30, activation = 'relu', name = 'Dense_30_Fourth'))
model_simple.add(Dense(5, activation = 'softmax', name = 'Output_5'))
model_simple.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Get the model selected:
def get_model(selected_NN):
    if selected_NN == 'Simple NN':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/Simple_model')
    elif selected_NN == 'Multi layers NN':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/Multi_model')
    elif selected_NN == 'Embedded Multi layers NN':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/Multi_model_Embbeded')
    elif selected_NN == 'Embbed Max pool Multi Layers NN':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/multi_model_Embed_Max')
    elif selected_NN == 'CNN Multi layers model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_Conv')
    return model

#Plot the model selected
model = get_model(selected_NN)
tf.keras.utils.plot_model(model, to_file='/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', show_shapes=True, show_layer_names=True)
st.sidebar.write('Summary of the model:')
st.sidebar.image('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', use_column_width = True)

#Get the history of the model selected:
def get_history(selected_NN):
    if selected_NN == 'Simple NN':
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/simple_model_history.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Multi layers NN':
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/multi_history.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embedded Multi layers NN':
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/multi_embed_history.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embbed Max pool Multi Layers NN':
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/multi_embed_max_history.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'CNN Multi layers model':
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/conv_history.npy', allow_pickle = 'TRUE').item()
    return history

history = get_history(selected_NN)

test = np.arange(1, len(history['accuracy'])+1)
fig = go.Figure(data = go.Scatter(x = test, y = history['accuracy']))
st.plotly_chart(fig)

fig = go.Figure(data = go.Scatter(x = test, y = history['val_accuracy']))
st.plotly_chart(fig)