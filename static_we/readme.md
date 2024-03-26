# Static Word Embeddings NN Model

## 1.) Static Embeddings
I used the GloVe from the torchtext module for static embeddings.
This is a module that contains pretrained word embeddings for our model to use. 

## 2.) Preprocessing
I started off using a preprocessing function that would remove punctuation and lemmatize.
The model proved to be better without this from the beginning so I removed it early on. 
I suspect that the punctuations provide insight to the models. For preparing our features,
I had to turn the sentences into indices of their tokens (with the nltk word_tokenizer) 
as found in GloVe. Both X sets (of train and test) are to be converted into arrays of their
embedding indices to passed into the model and processed. 

## 3.) Network Architectures
I experiemented with CNN, RNN, and GRU architectures.
I created a class for each that would receive the embedding matrix, number of target classes,
and appropriate parameters upon instantiation.
The CNN with frozen embeddings worked best as compared to the non-frozen
embeddings we did in class. This means that the static embeddings
were not adjusted in my final model. 

## 4.) Parameter Tuning
I tuned the parameters with a Grid Search approach after deciding that
CNN was receiving the best scores among the 3 NNs I trained.
The parameters to tune were: number of filters, dropout probability, kernel size, and 
the learning rate of the optimizer. This stage can be found in the 'training.ipynb' notebook.

## 5.) Model Evaluation
I evaluated my models using the f1 score and saved my model that received the best
of this across all my models tested. This can be found in 'saved_models/CNN.pth'

## Conclusions
* CNN proved to be the most effective model for this task.
* In this main script, I am training a CNN with frozen static embeddings.
* Optimized Parameters:
  1. kernel_sizes: [3,4,5]
  2. num_filters: 100 
  3. dropout_prob: 0.5 
  4. learning_rate of optimizer: 0.001
  5. training epochs: 100
* Accuracy: 0.7495, F1 score: 0.7535
