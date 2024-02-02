import joblib
import time
import streamlit as st
import pandas as pd
from src.function import *

st.set_page_config(page_title="Naive Bayes sentiment analysis", page_icon=":sunglasses:") 

# Initialize the DataFrame in session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['Input', 'Prediction', "class1 probability", "class2 probability"])

# Load / train the model
if 'model_type' not in st.session_state:
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
        "##### :blue[Select a model to continue]",
        key="model_type",
        options = ["movie review", "movie review small"],
        captions = ["POS/NEG", "Action/Romance"],
        index = None)

if st.session_state.model_type == None:
    st.error('Please select a model to continue')
else:
    st.success('model is ready to use!')

# Tabs --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tab1, tab2,tab3 = st.tabs(["test the model with a sentence", "run the test file", "explanation"])
with tab1:
    # Text input
    test_sentence = st.text_area(
        "##### :blue[Enter some text] 👇",
        placeholder = st.session_state.text_hint)   
        
    if st.session_state.model_type == None:
        st.error("Please select a model to continue")
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

# test the model --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("Test the model with a file",divider = "rainbow")

    test_type = st.radio("##### :blue[Select a dataset to test]",
                        options = ["movie review", "movie review small"],
                        captions = ["POS/NEG", "Action/Romance"],
                        index = None)

    if test_type == None:
        st.error('Please select a dataset to test the model')
    elif test_type == "movie review":
        test_file = 'src/movie_reviews/dev'
        st.session_state.model = joblib.load('src/model.pkl')
    elif test_type == "movie review small":
        test_file = 'src/movie_reviews_small/test'
        st.session_state.model = joblib.load('src/small_model.pkl')

    if test_type != None:
        # Load the test file
        progress_bar("Analysis in progess... Please wait", "We are done!")
        loaded_model = st.session_state.model
        results = loaded_model.system_test(test_file)
        confusion_matrix, Accuracy, precision, recall, F1 = loaded_model.evaluate(results)

        st.subheader("Evaluation metrics",divider = 'grey')
        st.metric(f"Total accuracy for **{test_type}**", f"{Accuracy*100:.1f}%")

        c1, c2, c3 = st.columns(3)
        for i, class_name in enumerate(loaded_model.class_dict):
            with c1:
                st.metric(f"Precision for **{class_name}**", f"{precision[i]*100:.1f}%")
            with c2:
                st.metric(f"Recall for **{class_name}**", f"{recall[i]*100:.1f}%")
            with c3:
                st.metric(f"F1 for **{class_name}**", f"{F1[i]*100:.1f}%")
        
        plot_confusion_matrix(confusion_matrix, loaded_model.class_dict)

        
# explanation --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("Knowledge background",divider = "rainbow")

    c1, c2 = st.columns([1.2,1])
    with c1:
        '''
        - **Bag of Words Model** This project uses a bag of words model to represent the text data. In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.
        - **Naive Bayes Classifier:** This is a probabilistic machine learning model that's used for classification. The crux of the classifier is the assumption of independence between every pair of features
        '''
    with c2:
        st.image("src/Naive_bayes.png",width=300,caption='Pseudo-code given in Figure 4.2 of the Jurafsky and Martin book')

    '''
    ## Naive Bayes Classifier in Python

    This script implements a Naive Bayes classifier for text classification.

    ### Class: NaiveBayes
    '''
    st.code('''class NaiveBayes():
        def __init__(self):
            self.class_dict = {}        # {class1: 0, class2: 1, ...}
            self.feature_dict = {}      # {feature1: 0, feature2: 1, ...}
            self.prior = None           # prior[x] = log(P(class))
            self.likelihood = None      # likelihood[x][y] = log(P(feature|class))''',language='python')

    '''
    The `NaiveBayes` class is the main class in this script. It has the following attributes:

    - `class_dict`: A dictionary mapping class names to integer indices.
    - `feature_dict`: A dictionary mapping feature names (words) to integer indices.
    - `prior`: A numpy array containing the log prior probabilities of each class.
    - `likelihood`: A 2D numpy array containing the log likelihoods of each feature given each class.

    '''
