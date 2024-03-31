# Fine-Tuned Transformers Model

## Task
* Fine-tuning of pretrained transformers models on huggingface 
to the rotten tomatoes dataset. 

## Parameter Optimization
* I created a function that receives a parameter grid and then trains 
each parameter combination until finally saving the combination with 
the best evaluation f1 score.
* [Parameter Tuning Colab Notebook](https://colab.research.google.com/drive/1YH_ob8S7ImbcLnSBWfye_a8P0uiBsRE1?authuser=6#scrollTo=_sdqnDO8XFFP)

## distilBERT:
* **Optimized Parameters:**
  * epochs: 5
  * learning_rate: 1e-05
  * batch_size: 16
  * weight_decay: 0.01
* *Test f1 score: 0.8433*

## roBERTa:
* **Optimized Parameters:**
  * epochs: 5
  * learning_rate: 2e-05
  * batch_size: 16
  * weight_decay: 0.01
* *Test f1 score: 0.878*

## deBERTa:
* **Optimized Parameters:**
  * epochs: 5
  * learning_rate: 2e-05
  * batch_size: 16
  * weight_decay: 0.01
* *Test f1 score: 0.9128*

## Conclusions:
**Process:**

* It was very difficult to tune these due to hardware issues. I used the colab GPUs which 
were very limited and as such my tunings for each model could only fit 2-4 combinations
of parameters. I had previously tried wider parameter grids but it would crash my sessions.

**Future Recommendations**
* For the future, I would recommend access to better GPUs for trying more models that are 
bigger. With that in mind, I would also recommend better tuning of each model by
widening the scope of parameters. Finally, to be more creative, again given that more GPU is
provided, one could combine models and experiment with mixed architectures and see how this would perform. 

**Final Summary:**
* After tuning the abovementioned models, the best turned out to be deBERTa. 
* In this main script, I am training deBERTa with the following parameters.
  1. learning_rate: 2e-05
  2. batch_size: 16 
  3. epochs: 5
  4. weight_decat: 0.01
* *Test f1 score: 0.9137*

