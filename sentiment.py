import joblib
import time
import streamlit as st
import pandas as pd
from function import *

st.set_page_config(page_title="Naive Bayes sentiment analysis", page_icon=":sunglasses:") 

# Initialize the DataFrame in session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Input', 'Prediction', "class1 probability", "class2 probability"])

# Load / train the model
if 'model' not in st.session_state:
    st.session_state.model = NaiveBayes()
    st.session_state.model_type = None
    st.session_state.text_hint = None

if st.session_state.model_type != None:
    st.session_state.text_hint = "Type something... I' ll try to categorize it"
    if st.session_state.model_type == "movie review":
        st.session_state.model = joblib.load('src/model.pkl')
    elif st.session_state.model_type == "movie review small":
        st.session_state.model = joblib.load('src/small_model.pkl')
else:
    st.session_state.text_hint = "Select a dataset to train the model"

# -------------- streaml frontend --------------
# Title
st.title("🎬 Sentiment Analysis")

# Train model
st.radio(
        "##### :blue[Select dataset to train model]",
        key="model_type",
        options = ["movie review", "movie review small"],
        captions = ["POS/NEG", "Action/Romance"],
        index = None)

if st.session_state.model_type == None:
    st.error('Please select a dataset to train the model')
else:
    st.success('model is ready to use!')

# Text input
test_sentence = st.text_area(
    "##### :blue[Enter some text] 👇",
    placeholder = st.session_state.text_hint)   
if st.session_state.model_type == None:
    st.error("Please select a dataset to train the model")
# -------------- backend -------------------
# Predict the class
if test_sentence.isalnum() and st.session_state.model_type != None:
    predicted_class_index, log_prob = st.session_state.model.test(test_sentence)
    predicted_class = class_name(predicted_class_index)

    # Add the input and prediction to the DataFrame
    new_index = len(st.session_state.df) + 1
    label = predicted_class.upper()
    st.session_state.df.loc[new_index] = [test_sentence, label, log_prob[1], log_prob[0]]
    
    # Progress bar
    progress_spin()
    st.markdown(f"##### This comment is {predicted_class}!")

# Display the DataFrame
st.dataframe(st.session_state.df, width=1000)
