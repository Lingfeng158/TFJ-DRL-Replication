# TFJ-DRL-Replication
A paper replication project for *Time-driven feature-aware jointly deep reinforcement learning (TFJ-DRL) for financial signal representation and algorithmic trading*. 

Algorithmic trading has become a hot topic since the adoption of computers in stock exchanges. There are two categories of algorithmic trading, one based on prior knowledge and another based on machine learning (ML). The latter is gaining more attentions these days, as comparing to methods based on prior knowledge, ML based methods does not require professional financial knowledge, research, or trading experience. 

However, there are several drawbacks to previous implementations of machine learning algorithmic trading:  

* Supervised learning methods are difficult to achieve online learning, due to the cost of training. They attempt to predict stock prices of the next time point, but accuracy of price prediction results in second error propagation during translation from price prediction to trading actions.

* Reinforcement learning (RL) methods lacks the ability to perceive and represent environment features, as well as the ability to dynamically consider past states and changing trends. 

The paper of interest (TFJ-DRL) aims to combine the strength from both deep learning and reinforcement learning by integrating Recurrent Neural Network (RNN) and policy gradient RL.

Through this document, I'm going to detail the steps I took and decisions I made to replicate the model TFJ-DRL. 

## Content Overview

This document is divided into 5 parts:

1. Data used in model and data acquisition
2. Data preprocessing
3. RNN model definition
4. Reinforcement model definition
5. Loss function design
6. Model training and weight selection
7. Model performance testing
8. Link to paper and other resources

## Data used in model and data acquisition

