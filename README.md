# Assignment 3 — Regularization and Stability

**Course:** Statistical Methods for Machine Learning  
**Student:** Laura Shulbayeva  
**Paper:** Bousquet & Elisseeff, Stability and Generalization, JMLR 2002

## Description

This project empirically studies the relationship between regularization strength, algorithmic stability, and generalization performance. The experiments are based on the paper by Bousquet & Elisseeff (2002) and implemented from scratch in Python.

## Files

- utils.py — ridge regression, lasso, least squares, stability metrics
- experiment1.py — main experiments on both datasets
- experiment2.py — effect of dataset size on stability
- experiment3.py — L1 vs L2 regularization comparison
- main.py — runs all experiments
- plots/ — generated figures

## Datasets

- Synthetic: generated with make_regression, 250 samples, 10 features, gaussian noise
- Diabetes: sklearn dataset, 442 samples, 10 features, continuous regression target

## How to run

pip install numpy matplotlib scikit-learn
python main.py

## Declaration

I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.