![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Exercise:** Which of the models is the most accurate (SVM or Kover)? Can you guess why?

In this case, kover learns the most accurate model. We can see that the training accuracy of SVM is significantly greater than its testing accuracy, so it seems to be overfitting. In fact, methods that try to learn sparse models (using very few variables), such as Kover, are generally more successful in avoiding overfitting than those that don't (e.g.: SVM).
