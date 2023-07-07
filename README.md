# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
We focus on Federated Learning in Semantic Segmentation, using Idda and GTAV datasets.
In particular we move from a centralized scenario to a realistic federated and unsupervised scenario in the following phases:
### Step 1
Centralized approach, training and testing on Idda.
### Step 2
Federated approach, training and testing on Idda.
### Step 3
Pretraining a model on GTAV, testing on Idda.
### Step 4
Using pre-trained model from previous step, we generate pseudo label in unsupervised learning to improve model's performances.
We train on unlabeled  Idda, testing on Idda.
### Step 5
We propose a clustering-based version, in which there are specific classifiers for clusters. Clustering, training and testing on Idda.
### Losses
We propose different ways to reduce cross entropy loss: simple mean, mean between class averages, one inversely proportional to the frequency, and one based on weights.
### Implemantion
Please see at Demo.ipynb to see how to replicate our experiments.
