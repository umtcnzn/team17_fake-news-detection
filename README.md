# Fake News Detection Project

This project aims to classify news articles as *true* or *fake* using both classical machine learning models and a Recurrent Neural Network (RNN) with LSTM.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Models Used](#models-used)

## Introduction
The goal of this project is to build a system capable of detecting fake news articles. The project leverages both classical machine learning algorithms and deep learning techniques to achieve robust classification.

## Dataset
The dataset used in this project consists of **40,000** news articles: **20,000 true news** and **20,000 fake news**. The data was processed using Natural Language Processing (NLP) techniques, including vectorization. Here in this link, you can reach the dataset that we used in this project. 
https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

## Models Used
The following models were used in this project:
1. **Logistic Regression**: A simple, yet effective model for binary classification tasks.
2. **Decision Tree**: A tree-based model that makes decisions based on feature splits.
3. **Random Forest**: An ensemble of Decision Trees, which improves accuracy by reducing overfitting.
4. **Gradient Boosting**: A boosting technique that builds models sequentially to improve predictions.
5. **LSTM (Long Short-Term Memory)**: A type of RNN that captures sequential dependencies in text data, making it suitable for NLP tasks like fake news detection.

