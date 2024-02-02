import time
import io
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def progress_bar(text0,text):
    progress_text = text0
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

    def system_test(self, dev_set):
        results = defaultdict(dict)

        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                if name == '.DS_Store':
                    continue
                with open(os.path.join(root, name)) as f:
                    correct_class = root.split('/')[-1]
                    results[name]['correct'] = self.class_dict[correct_class]
                    feature_vector = np.zeros(len(self.feature_dict)) 
                    for line in f: 
                        for word in line.split(): 
                            if word in self.feature_dict:
                                feature_vector[self.feature_dict[word]] += 1
                    log_likelihoods = np.dot(feature_vector, self.likelihood) 
                    log_prob = log_likelihoods + self.prior
                    predicted_class = np.argmax(log_prob)
                    results[name]['predicted'] = predicted_class
        return results
    
    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        Accuracy = 0
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)))

        # Calculate the confusion matrix and accuracy
        for filename in results:
            correct_class = results[filename]['correct']
            predicted_class = results[filename]['predicted']
            if correct_class == predicted_class:
                Accuracy += 1
            confusion_matrix[correct_class][predicted_class] += 1
        
        # Initialize the metrics for all classes
        num_of_classes = len(self.class_dict)
        precision = np.zeros(num_of_classes)
        recall = np.zeros(num_of_classes)
        F1 = np.zeros(num_of_classes)

        # Calculate the precision, recall and F1 for each class
        for i in self.class_dict.values():
            if np.sum(confusion_matrix, axis=0)[i] != 0:
                precision[i] =  confusion_matrix[i][i] / np.sum(confusion_matrix, axis=0)[i]
            else:
                precision[i] = 0
            if np.sum(confusion_matrix, axis=1)[i] != 0:
                recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix, axis=1)[i]
            else:
                recall[i] = 0
            if precision[i] != 0 or recall[i] != 0:
                F1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:    
                F1[i] = 0
        
        if len(results) != 0:
            Accuracy /= len(results)
        else:
            Accuracy = 0

        return confusion_matrix, Accuracy, precision, recall, F1
        print("\n",self.class_dict)
        print("\nAccuracy is :", Accuracy)
        for i in self.class_dict.values():
            print("-----------------------")
            print("precision for ", i, " is :", precision[i])
            print("recall for ", i, " is :", recall[i])
            print("F1 for ", i, " is :", F1[i])

def plot_confusion_matrix(confusion_matrix, class_dict):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='g',
                xticklabels=class_dict.keys(), 
                yticklabels=class_dict.keys(), 
                cmap='YlGnBu')
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Display the plot in Streamlit
    st.image(buf, caption='Confusion Matrix', use_column_width=True)

# ### Method: train

# The `train` method trains the Naive Bayes classifier on a training set. It fills in the `prior` and `likelihood` attributes based on the training data.

# ### Method: test

# The `test` method tests the classifier on a development or test set. It returns a dictionary of filenames mapped to their correct and predicted classes.

# ### Method: evaluate

# The `evaluate` method calculates and prints the precision, recall, F1 score for each class, and overall accuracy given the results from the `test` method.
