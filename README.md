# -CLASSICAL MACHINE LEARNING-

I tried using different datasets to enrich the training set. 
This includes the IMDB sentiment dataset and twitter dataset.
The twitter dataset was very noisy and large so I tried using random 
subsets of a few thousand rows but it never improved the scores. 
Ultimately, better results were using only the train set of the IMDB dataset.

I tested various classifier models such as: Logistic Regression, Stochastic
Gradient Descent, Naive Bayes, and XGBoost. I also created my own preprocessor
although scores were higher without preprocessing so I did not continue to use it. 
Probably, the punctuation marks and capitalizations help the algorithm understand
the emotions from the reviews. 

Hyperparameter Tuning using Bayes Search Cross Validation was done 
Strangely however, single fitting of default parameters achieved better scores 
was faster to fit and test and so my final model does just this.

## Final Model: 
1. Logistic Regression
   1. C=1, l2 penalty, saga solver, everything else set to default 
2. Train Set + IMDB Train Set, 
3. No preprocessing.
* Final f1 Score on Test Set: 81.64%

## Notes
* I included a video of my final sample test run before submitting this directory. (working_demo_test.mov)
* I included two models that I saved of succesful logistic regression model runs.
* My original workspace where I tested multiple models and data enrichments can 
be found in this Colab Notebook link.
  * https://colab.research.google.com/drive/1rmpWX95Onf1C_HjHN9Dy2ez3IWO_Mcny?usp=sharing

## Testing
* cd francisco_alba_individual_project
* python3 main.py