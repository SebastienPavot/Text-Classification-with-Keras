#Libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

#Options
st.set_option('deprecation.showPyplotGlobalUse', False)
st.beta_set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title="Text Classification SPavot",  # String or None. Strings get appended with "• Streamlit". 
	page_icon="😷",  # String, anything supported by st.image, or None.
)

#Data importation
train = pd.read_csv('/Users/spavot/Documents/Perso/Text classification & Visualization/Data/train_cleaned.csv')
test = pd.read_csv('/Users/spavot/Documents/Perso/Text classification & Visualization/Data/test_cleaned.csv')

#Title
st.title('Text classification with Keras')

#Overview of the dataset:
st.write("""
The original data: Covid-19 tweets related
""")

#Shape and head:
st.write('Shape of train dataset:', train.shape)
st.write('Shape of test set:', test.shape)

#Select the model=
selected_NN = st.sidebar.selectbox("Select the Neural Network", ('Simple NN', 'Multi layers NN', 'Embedded Multi layers NN', 'Embbed Max pool Multi Layers NN', 
                                    'CNN Multi layers model'))

#Get the values of the input
train_text = train.OriginalTweet.values
test_text = test.OriginalTweet.values
#Dummy encode the target
y_train = pd.get_dummies(train.Sentiment).values
y_test = pd.get_dummies(test.Sentiment).values
#Initate the label of our target:
CATEGORIES = ["Neutral", "Positive", "Extremely Negative", 'Negative', 'Extremely Positive']
#Initiate maxlen for vocab:
maxlen = 30
#Initiate vectorizer and tokenizer regardng the model selected:
vectorizer = CountVectorizer()
vectorizer.fit(train_text)
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(train_text)
#Initiate the name of the models into an array:
models_name = ['Simple NN', 'Multi layers NN', 'Embedded Multi layers NN', 'Embbed Max pool Multi Layers NN', 'CNN Multi layers model']

#Functions part:

#Get the training and test sets in the shape of the models needs:
def set_training_tests(selected_NN):
    #Input data regarding the model selected
    if selected_NN == 'Simple NN' or selected_NN == 'Multi layers NN':
        #Create training / test set:
        X_train = vectorizer.transform(train_text)
        X_test  = vectorizer.transform(test_text)
    else:
        #Fit the tokenizet to the data
        X_train = tokenizer.texts_to_sequences(train_text)
        X_test = tokenizer.texts_to_sequences(test_text)
        vocab_size = len(tokenizer.word_index)+1
        #Pad the text
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test

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

#Definition of the model selected:
simple = 'It is a simple one layer model without any specifics optimizations'
multi = 'Simple model but deeper with 3 layers'
Embbed = 'We applied word embedding and add an embedded layer'
Max_pool = 'Using word embedding, this time we add a max layers to avoid the overfitting'
Conv = 'This model uses word embedding with one convulotionnal layer'

#Function to assign description regarding the selected model:
def model_description(selected_NN):
    if selected_NN == 'Simple NN':
        description = simple
    elif selected_NN == 'Multi layers NN':
        description = multi
    elif selected_NN == 'Embedded Multi layers NN':
        description = Embbed
    elif selected_NN == 'Embbed Max pool Multi Layers NN':
        description = Max_pool
    elif selected_NN == 'CNN Multi layers model':
        description = Conv
    return description


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

#Return the prediction based on the input and the model selected
def prediction_input_user(user_input):
    #Transform the input into a numpy array
    user_input = [user_input]
    user_input = np.array(user_input)
    #Transform the array in the shape the model selected need:
    if selected_NN == 'Simple NN' or selected_NN == 'Multi layers NN':
        user_input = vectorizer.transform(user_input)
    else:
        user_input = tokenizer.texts_to_sequences(user_input)
        user_input = pad_sequences(user_input, padding='post', maxlen=maxlen)
    #Predict using the model selected
    prediction = model.predict(user_input)
    #Return the label predicted
    pred_name = CATEGORIES[np.argmax(prediction)]
    return pred_name

#Design part:

#Plot the model selected
model = get_model(selected_NN)
tf.keras.utils.plot_model(model, to_file='/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', show_shapes=True, show_layer_names=True)
st.sidebar.write('Summary of the model:')
st.sidebar.image('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', use_column_width = True)

#Assignt the description of the model selected
description = model_description(selected_NN)
st.sidebar.write('Model description:', description)

#Accuracy and loss plots:
#Get the history
history = get_history(selected_NN)
#Initiate an array of range of number of epochs
test = np.arange(1, len(history['accuracy'])+1)

#Loss plot
fig = go.Figure()
fig.add_trace(go.Scatter(name = 'Training set', x = test, y = history['loss']))
fig.add_trace(go.Scatter(name = 'Test set', x = test, y = history['val_loss']))
fig.update_layout(
    title="Training & Validation loss evolution",
    xaxis_title="Epochs",
    yaxis_title="Loss",
    legend_title="Train & Test sets performance",
    width = 1250)
st.plotly_chart(fig)

# Accuracy plot
fig = go.Figure()
fig.add_trace(go.Scatter(name = 'Training set', x = test, y = history['accuracy']))
fig.add_trace(go.Scatter(name = 'Test set', x = test, y = history['val_accuracy']))
fig.update_layout(
    title="Training & Validation accuracy evolution",
    xaxis_title="Epochs",
    yaxis_title="Accuracy",
    legend_title="Train & Test sets performance",
    width = 1250)
st.plotly_chart(fig)

#Confusion matrix plot:
#Get X_train, X_test
X_train, X_test = set_training_tests(selected_NN)
#Get the prediction of the model on the test set:
prediction = model.predict(X_test)
#Get one for the highest prediction, else 0:
for i in range(0,len(prediction)):
    for y in range(0,5):
        if prediction[i][y] == max(prediction[i]):
            prediction[i][y] = 1
        else:
            prediction[i][y] = 0

#Compute the confusion matrix:
conf_mat = confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))
#Plot it:
plt.figure(figsize=(25,7))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.ylabel('Actual', fontsize=15)
plt.xlabel('Predicted', fontsize=15)
plt.title('Confusion matrix of the predictions on the test set:', fontsize=20)
st.pyplot()

#Prediction of user inputted data:
user_input = st.text_input("Challenge the model, input your own tweet!", 'Try it yourself :)')
pred_name = prediction_input_user(user_input)
st.write("The model think that it is:", pred_name)

#Credit and social link:s
st.sidebar.write('Made by Sébastien PAVOT: https://github.com/SebastienPavot')

#Allow the possibility to see predictions of the others models:
Others_Pred =  st.checkbox('See what other models think:')

#If the checkbox is ticked:
if Others_Pred:
    Model_Name = []
    Models_Prediction = []
    #For each model, we predct using the prediction function we created before and append the prediction / model name to an array
    for i in models_name:
        Model_Name.append(i)
        Models_Prediction.append(prediction_input_user(user_input))
    #Transform the arrays into dataframe
    Predictions_All = pd.DataFrame(Model_Name, columns=['Model Name'])
    Predictions_All['Prediction'] = Models_Prediction
    #Output the dataframe
    st.write('What does the others think 🤔', Predictions_All)