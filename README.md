# team17_fake-news-detection
# Fake News Detection Project
This project aims to classify news articles as true or fake using both classical machine learning models and a Recurrent Neural Network (RNN) with LSTM.

Table of Contents
Introduction
Dataset
Models Used
Project Structure
How to Run
Results
Introduction
The goal of this project is to build a system capable of detecting fake news articles. The project leverages both classical machine learning algorithms and deep learning techniques to achieve robust classification.

Dataset
The dataset used in this project consists of 40,000 news articles: 20,000 true news and 20,000 fake news. The data was processed using Natural Language Processing (NLP) techniques, including vectorization.

Models Used
The following models were used in this project:

Logistic Regression: A simple, yet effective model for binary classification tasks.
Decision Tree: A tree-based model that makes decisions based on feature splits.
Random Forest: An ensemble of Decision Trees, which improves accuracy by reducing overfitting.
Gradient Boosting: A boosting technique that builds models sequentially to improve predictions.
LSTM (Long Short-Term Memory): A type of RNN that captures sequential dependencies in text data, making it suitable for NLP tasks like fake news detection.
