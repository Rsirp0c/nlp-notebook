import time
import streamlit as st

def progress_bar(text):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success(text)

def progress_spin():
    with st.spinner('Wait for it...'):
        time.sleep(1.5)

def class_name(class_index):
        return [class_name for class_name, index in st.session_state.model.class_dict.items() if index == class_index][0]


# Naive Bayes Classifier   
import os
import joblib
import numpy as np
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        self.class_dict = {}        # {class1: 0, class2: 1, ...}
        self.feature_dict = {}      # {feature1: 0, feature2: 1, ...}
        self.prior = None           # prior[x] = log(P(class))
        self.likelihood = None      # likelihood[x][y] = log(P(feature|class))

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[feature, class] = log(P(feature|class))
    '''
    def train(self, train_set):
        #trace the count of classes and features
        class_count = defaultdict(int)     
        feature_count = defaultdict(lambda: defaultdict(int))

        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:  #, "r", encoding="utf8"
                    # count classes
                    class_name = root.split('/')[-1]
                    if class_name not in self.class_dict:
                        self.class_dict[class_name] = len(self.class_dict)
                    class_count[class_name] += 1                  
                    # count features
                    for line in f:
                        for word in line.split():
                            if word not in self.feature_dict:
                                self.feature_dict[word] = len(self.feature_dict)
                            feature_count[class_name][word] += 1
            
        # initialize the numpy matrix using numpy.zeros
        self.prior = np.zeros(len(self.class_dict))
        self.likelihood = np.zeros((len(self.feature_dict),len(self.class_dict)))
        total_classes = sum(class_count.values())

        for class_name, count in class_count.items():
            # Calculate class prior
            class_index = self.class_dict[class_name]
            self.prior[class_index] = np.log((count / total_classes))

            # Calculate likelihood
            total_words = sum(feature_count[class_name].values()) + len(self.feature_dict)
            for feature_name, feature_index in self.feature_dict.items():
                count = feature_count[class_name][feature_name] + 1  
                self.likelihood[feature_index][class_index] = np.log(count / total_words)

    '''         
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, sentence):
        processed_sentence = sentence.lower().split()

        # Create a feature vector
        feature_vector = np.zeros(len(self.feature_dict))
        for word in processed_sentence:
            if word in self.feature_dict:
                feature_vector[self.feature_dict[word]] += 1

        # Predict
        log_likelihoods = np.dot(feature_vector, self.likelihood)
        log_prob = log_likelihoods + self.prior
        return np.argmax(log_prob),log_prob