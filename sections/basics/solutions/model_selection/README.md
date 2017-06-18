![#c5f015](https://placehold.it/15/c5f015/000000?text=+) **Exercise:** Train a [RBF kernel Support Vector Machine](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifier on the [breast cancer](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) data set from the UCI repository. To do so, run the following command.

```bash
make basics.model.selection
```

1. Is the accuracy on the training set a good estimator of the accuracy on the testing set?

No, we can clearly see that the hyperparameters that lead to the best accuracy on the training set are not the same as the ones that lead to the best accuracy on the testing set.

2. Given the information at hand, which combination of hyperparameters should you choose? Why?

We should choose the following hyperparameter values: *SVM C = 2* and *RBF kernel gamma = 7*, since they lead to the greatest cross-validation accuracy.

3. Do some hyperparameter combinations lead to overfitting? Give an example.

Yes, *SVM C = 9* and *RBF kernel gamma = 9* lead to overfitting, since the training accuracy is 100%, but the testing accuracy is 51%. 
This is the accuracy that would be achieved by a random predictor (assuming that the classes are balanced). 
Many other hyperparameter values (in the bottom right corner) also lead to overfitting.

4. Do some hyperparameter combinations lead to underfitting? Give an example.

Yes, *SVM C = 0* and *RBF kernel gamma = 0* lead to overfitting, since the training accuracy is 66%, but the testing accuracy is 51%. Many other hyperparameter values (in the bottom right corner) also lead to underfitting.
