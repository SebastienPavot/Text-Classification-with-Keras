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
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

#Download a list of stopwords
stop_words = set(stopwords.words('english'))
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
#Options to either see at a model or overall level
st.sidebar.write('See the model performances or pick one to have more details and play with it:')
models_details = st.sidebar.checkbox('Focus on a model',False)
#Get the values of the input
train_text = train.OriginalTweet.values
test_text = test.OriginalTweet.values
#Dummy encode the target
y_train = pd.get_dummies(train.Sentiment).values
y_test = pd.get_dummies(test.Sentiment).values
#Initate the label of our target:
CATEGORIES = ["Neutral", "Positive", 'Negative']
#Initiate maxlen for vocab:
maxlen = 30
#Initiate vectorizer and tokenizer regardng the model selected:
vectorizer = CountVectorizer()
vectorizer.fit(train_text)
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(train_text)
#Initiate the name of the models into an array:
models_name = ['CountVectorizer One layer Model', 'CountVectorizer Multi layers Model', 'Embedding One layer Model', 'Embedding Multi layers Model', 
                                    'Embedding Glove One layer Model', 'Embedding Glove Multi layers Model', 'Convolutional Model', 'Convolutional Glove Model']

#Functions part:
#Get the training and test sets in the shape of the models needs:
def set_training_tests(selected_NN):
    #Input data regarding the model selected
    if selected_NN == 'CountVectorizer One layer Model' or selected_NN == 'CountVectorizer Multi layers Model':
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

#Get the model selected, history related and the description:
def get_model(selected_NN):
    if selected_NN == 'CountVectorizer One layer Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/Simple_model_Count.h5')
        description = 'One layer model using CountVectorizer to encode the text.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_count.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'CountVectorizer Multi layers Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_multi_count.h5')
        description = 'Multi layers model using CountVectorizer to encode the text.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_multi_count.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embedding One layer Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_simple_embed.h5')
        description = 'One layer model using an embedding layer, tokenizer and pad sequences from Keras.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_embed.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embedding Multi layers Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/multi_model_Embed.h5')
        description = 'Multi layers model using an embedding layer, tokenizer and pad sequences from Keras.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_multi_Embed.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embedding Glove One layer Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_simple_glove.h5')
        description = 'One layer model using an embedding layer with Glove dictionnary as weights, tokenizer and pad sequences from Keras.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_glove.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Embedding Glove Multi layers Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_multi_glove.h5')
        description = 'Multi layers model using an embedding layer with Glove dictionnary as weights, tokenizer and pad sequences from Keras.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Multi_glove.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Convolutional Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_Conv.h5')
        description = 'Convolutional model using embedding layer and convolutional layer.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Conv.npy', allow_pickle = 'TRUE').item()
    elif selected_NN == 'Convolutional Glove Model':
        model = tf.keras.models.load_model('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/NN_Models/model_Conv_glove.h5')
        description = 'Convolutional model using embedding layer with Glove dictionnary as weights and convolutional layer.'
        history = np.load('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Conv_glove.npy', allow_pickle = 'TRUE').item()
    return model, description, history

#Function to clean the user input to make it as the model learned
def clean_user_input(user_input):
    user_input_cleaned = user_input.lower()
    user_input_cleaned = user_input_cleaned.replace('http\S+|www.\S+', '')
    user_input_cleaned = user_input_cleaned.replace('#', '')
    user_input_cleaned = ' '.join([word for word in user_input_cleaned.split() if word[0] != '#'])
    user_input_cleaned = user_input_cleaned.replace(r'[^\w\s]', '')
    user_input_cleaned = ' '.join([word for word in user_input_cleaned.split() if word not in (stop_words)])
    user_input_cleaned = ' '.join([word for word in user_input_cleaned.split() if word.isalpha()])
    lemmatizer = WordNetLemmatizer()
    user_input_cleaned = ' '.join([lemmatizer.lemmatize(word) for word in user_input_cleaned.split()])
    return user_input_cleaned

#Return the prediction based on the input and the model selected
def prediction_input_user(user_input_cleaned):
    #Transform the input into a numpy array
    user_input = [user_input_cleaned]
    user_input = np.array(user_input)
    #Transform the array in the shape the model selected need:
    if selected_NN == 'CountVectorizer One layer Model' or selected_NN == 'CountVectorizer Multi layers Model':
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
#If model details:
if models_details:
    #Select the model=
    selected_NN = st.sidebar.selectbox("Select the Neural Network", ('CountVectorizer One layer Model', 'CountVectorizer Multi layers Model', 'Embedding One layer Model', 'Embedding Multi layers Model', 
                                    'Embedding Glove One layer Model', 'Embedding Glove Multi layers Model', 'Convolutional Model', 'Convolutional Glove Model'))
    #Plot the model selected and the description:
    model, description, history = get_model(selected_NN)
    tf.keras.utils.plot_model(model, to_file='/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', show_shapes=True, show_layer_names=True)
    st.sidebar.write('Summary of the model:')
    st.sidebar.image('/Users/spavot/Documents/Perso/Text classification & Visualization/Models/Plot/model_plot.png', use_column_width = True)
    #Assignt the description of the model selected
    st.sidebar.write('Model description:', description)
    #Accuracy and loss plots:
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
        for y in range(0,2):
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
    pred_name = prediction_input_user(clean_user_input(user_input))
    st.write("The model think that it is:", pred_name)
    #Show the sentence after cleaning
    st.write('The tweet after cleaning that we will give to the model:', clean_user_input(user_input))

    #Credit and social link:s
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
#If we want to see at overall level
else:
    #First first all the models into a dataframe in the side bar
    st.sidebar.write('All the models:')
    models_name_df = pd.DataFrame(models_name, columns = ['Models'])
    st.sidebar.table(models_name_df)

    #Initate an array to iterate over the models history paths to get it
    list_history = ['/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_count.npy', '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_multi_count.npy', \
                    '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_embed.npy', '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_multi_Embed.npy', \
                    '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_simple_glove.npy', '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Multi_glove.npy', \
                    '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Conv.npy', '/Users/spavot/Documents/Perso/Text classification & Visualization/Models/History/history_Conv_glove.npy'] 

    #Initiate a plotly figure
    fig = go.Figure()
    #For each model, add the loss history as a line
    for i,y in zip(list_history, models_name):
        history = np.load(i, allow_pickle = 'TRUE').item()
        test = np.arange(1, len(history['accuracy'])+1)
        fig.add_trace(go.Scatter(name = y, x = test, y = history['loss']))
    #Add details and legend
    fig.update_layout(
    title="Training loss evolution",
    xaxis_title="Epochs",
    yaxis_title="Loss",
    legend_title="Training set performance",
        width = 1250)
    st.plotly_chart(fig)

    fig = go.Figure()
    #For each model, add the validation loss history as a line
    for i,y in zip(list_history, models_name):
        history = np.load(i, allow_pickle = 'TRUE').item()
        test = np.arange(1, len(history['accuracy'])+1)
        fig.add_trace(go.Scatter(name = y, x = test, y = history['val_loss']))
    
    fig.update_layout(
    title="Validation loss evolution",
    xaxis_title="Epochs",
    yaxis_title="Loss",
    legend_title="Test set performance",
        width = 1250)
    st.plotly_chart(fig)

    fig = go.Figure()
    #For each model, add the accuracy history as a line
    for i,y in zip(list_history, models_name):
        history = np.load(i, allow_pickle = 'TRUE').item()
        test = np.arange(1, len(history['accuracy'])+1)
        fig.add_trace(go.Scatter(name = y, x = test, y = history['accuracy']))
    
    fig.update_layout(
    title="Training accuracy evolution",
    xaxis_title="Epochs",
    yaxis_title="Accuracy",
    legend_title="Training set performance",
        width = 1250)
    st.plotly_chart(fig)

    fig = go.Figure()
    #For each model, add the validation accuracy history as a line
    for i,y in zip(list_history, models_name):
        history = np.load(i, allow_pickle = 'TRUE').item()
        test = np.arange(1, len(history['accuracy'])+1)
        fig.add_trace(go.Scatter(name = y, x = test, y = history['val_accuracy']))
    
    fig.update_layout(
    title="Validation accuracy evolution",
    xaxis_title="Epochs",
    yaxis_title="Accuracy",
    legend_title="Test set performance",
        width = 1250)
    st.plotly_chart(fig)

#In any case, at github link of the project at the end of the side bar
st.sidebar.write('Made by Sébastien PAVOT: https://github.com/SebastienPavot')
