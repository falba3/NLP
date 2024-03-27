# Fine-Tuned Transformers Model

## Task
* Fine-tuning of pretrained transformers models on huggingface 
to the rotten tomatoes dataset. 

## Parameter Optimization
* I created a function that receives a parameter grid and then trains 
each parameter combination until finally saving the combination with 
the best evaluation f1 score.
* [Parameter Tuning Colab Notebook](https://colab.research.google.com/drive/1vtNBEbhre3c0S_qnfLrE6SJ96LSM6jkd?usp=sharing)

## DistilBERT
* epochs: 5
* learning_rate: 1e-05
* batch_size: 16
* weight_decay: 0.01
* *Test f1 score: 0.8386*

## deBERTa
* epochs: 1
* learning_rate: 1e-05
* batch_size: 16
* weight_decay: 0.01
* *Test f1 score: 0.9118*


## Conclusions
* After tuning multiple models, the best turned out to be ____
* In this main script, I am training ____ with the following parameters.
* Optimized Parameters:
  1. learning_rate: 
  2. batch_size:  
  3. epochs: 
  4. weight_decat:
* Test f1 score: 0.9118
