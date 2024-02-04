CLASSICAL MACHINE LEARNING

I tried using different datasets to enrich the training set. 
This includes the IMDB sentiment dataset and twitter dataset.
Ultimately, better results were using only the train set of the IMDB dataset.

I tested various classifier models such as: Logistic Regression, Stochastic
Gradient Descent, Naive Bayes, and XGBoost. I also created my own preprocessor
although scores were higher without preprocessing so I did not continue to use it.

Hyperparameter Tuning using Bayes Search Cross Validation was done however, 
single fitting of default parameters achieved better scores with faster performance.

Final Model: 
Logistic Regression, Default Parameters, Train Set + IMDB Train, No preprocessing.