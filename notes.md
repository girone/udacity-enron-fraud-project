# Notes on the project

## Preparation

### Available features (as described at the end of the PDF)

* Payments
  * Director fees
    Keiner der POIs hat hier einen nicht-leeren Eintrag.
  * Insgesammt ist der relative Anteil jeder payment-Kategorie am jeweiligen total payment wahrscheinlich aussagekräftiger, als absolute Werte.
* Stock value
  * Viele POIs haben einen hohen total stock value
* Email features
    I doubt that the email features (from_this_person_to_poi, from_poi_to_this_person) are admissible as given. This is because when splitting the data into training and test set, we assume the labels of the test set are unknown. However, the feature counts emails to all pois, not only those in the training set.

### Feature selection

Check individual features on their importance as done in [this example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html). I guess it is unlikely that there is a clean winner, but never mind trying.

### Ideas for new features

1. Emails from and to POIs as shares.
2. Set every payment and stock feature in relation to the total (maybe this can be achieved with PCA without adding N new features).
3. Text features. Think about some more complpex metric here.
4. Ratio of total payments to total stock.
5. Relativate all financial features to the respective total..

### Thoughts on algorithms

We have labelled data, with discrete labels. Thus, use supervised learning (from the lecture: Decision tree, naive Bayes, SVM, ensemble methods (Forest of random trees), k-nearest neighbors, LDA, logistic regression; or other sklearn algorithms: Stochastic Gradient Descent SGD, Gaussian Process Classification GPC, ensemble methods(AdaBoost, Gradient Tree Boosting), Neutral Networks (Multi-Layer Perceptron MLPClassifier)).

## Experiments

### 2018-05-31

First promising results: LinearSVC + feature "shared_receipt_with_poi" and the two new email features. Compared to LinearSVC + all features this is similar at first glance.

#### Next steps (1)

1. Look into the features, visualize them.
2. If there are outliers, think about how to remove them. Only if the results don't get much better think about new features.
3. Feature scaling.

### 2018-06-01

Wrote simple code to visualize features including annotation. This immediately showed that the entry "TOTAL" is an outlier in the dataset. Removed it from the data. Repeated the parameter sets from yesterday, they showed slightly better results.

Visualized the remaining data by plotting each variable against the others. Findings:

* deferred_income seems to have negative range
* loan advances has one high outlier
* salary and director_fees are mutally exclusive (either one is positive or the other)
  * POIs dont have director_fees
* total_payments has one outlier

Redo it with POIs highlighted. Include new features.

Some new results now: NaiveBayes outperforms LinearSVC.

GaussianNB(default parameters)
    Accuracy: 0.74713    Precision: 0.23578    Recall: 0.40000    F1: 0.29668    F2: 0.35109
    Total predictions: 15000    True positives:  800    False positives: 2593    False negatives: 1200    True negatives: 10407

LinearSVC(default parameters)
    Accuracy: 0.74087    Precision: 0.21208    Recall: 0.34750    F1: 0.26341    F2: 0.30815
    Total predictions: 15000    True positives:  695    False positives: 2582    False negatives: 1305    True negatives: 10418

#### Next steps (2)

1. Remove outliers systematically.
2. Do some fine-tuning, some scaling.
3. Extract text features: Top words by phrases, create clusters, see if there are more frequent terms for POIs.

### 2018-06-02 (1)

Read about outlier detection with sklearn. LOF seems to be fitting for the present task and data set. Implemented it, using the featureFormat function given used by the tester code. Had problems with a different array lenght of the outlier labels and the keys in the data. Solved it by using `featureFormat()` in a way that NaN is translated to 0.0, but entries with all 0.0 values are kept.

I use all available features (not the two computed email features) for the outlier detection. For the results I compare Naive Bayes and SVC with the default parameters. _Note_ that the nearest neighbor distance works best with equidistant dimensions, which might not be true for all features. Some feature scaling could improve the results. However, I feel for now this is good enough.

#### 10% outliers

Changed results:

GaussianNB(default parameters)
    Accuracy: 0.37792    Precision: 0.09719    Recall: 0.85500    F1: 0.17454    F2: 0.33406
    Total predictions: 13000    True positives:  855    False positives: 7942    False negatives:  145    True negatives: 4058

LinearSVC(default parameters)
    Accuracy: 0.73662    Precision: 0.12149    Recall: 0.38900    F1: 0.18515    F2: 0.27006
    Total predictions: 13000    True positives:  389    False positives: 2813    False negatives:  611    True negatives: 9187

Gaussian Naive Bayes has worse accuracy, poor precision but good recall now. Maybe too many outliers have been removed (5 out of 15 outliers are POIs).

#### 5% outliers

Removing the 5% outliers found this way, we remove 8 outliers (3 of which are POIs, which represents an even larger share). The Naive Bayes's precision gets better, and SVCs precision also looks promising now, while recall does not change much. Time to work on the algorithm parameters.

GaussianNB(default parameters)
    Accuracy: 0.33579    Precision: 0.14979    Recall: 0.78050    F1: 0.25135    F2: 0.42370
    Total predictions: 14000    True positives: 1561    False positives: 8860    False negatives:  439    True negatives: 3140

LinearSVC(default parameters)
    Accuracy: 0.72829    Precision: 0.22667    Recall: 0.37400    F1: 0.28226    F2: 0.33097
    Total predictions: 14000    True positives:  748    False positives: 2552    False negatives: 1252    True negatives: 9448

#### Next steps (3)

1. Do feature scaling prior to outlier removal.
2. Try other algorithms and do some parameter tuning.
3. Try to extract the aforementioned email features.

### 2018-06-02 (2)

Use `scale()` and `robust_scale()` to scale features. Might improve results of SVC, not Naive Bayes.

Results got quite a lot better, also less POIs have been removed as outliers. Thus, also for Naive Bayes the results improved (by a large extend, actually). However, applying it seemed too easy, so I need to find a way to validate my code.

Added argparse to the `poi_id.py` script, to control the preprocessing and used algorithm without changing the code. Need to set the default to the values I choose for the submission when cleaning up.

One more change: Remove the `"TOTAL"` entry before feature scaling, to reduce its bias on the data. Indeed, this does not change the results when using `robust_scale()`, but it seems fair to remove it before `scale()`.

Observation during validation: NaN values become numbers. They are probably treated as zeros in the input. This seems to be valid, because it's is just what the tester code does. The validation shows no problems so far. So here are the next results:

#### Preprocess with `scale()`

GaussianNB(default parameters)
    Accuracy: 0.34179    Precision: 0.14552    Recall: 0.74050    F1: 0.24325    F2: 0.40738
    Total predictions: 14000    True positives: 1481    False positives: 8696    False negatives:  519    True negatives: 3304

LinearSVC(default parameters)
    Accuracy: 0.83036    Precision: 0.33480    Recall: 0.19000    F1: 0.24242    F2: 0.20799
    Total predictions: 14000    True positives:  380    False positives:  755    False negatives: 1620    True negatives: 11245

#### Preprocess with `robust_scale()`

GaussianNB(default parameters)
    Accuracy: 0.31907    Precision: 0.17211    Recall: 0.98850    F1: 0.29317    F2: 0.50726
    Total predictions: 14000    True positives: 1977    False positives: 9510    False negatives:   23    True negatives: 2490

LinearSVC(default parameters)
    Accuracy: 0.80521    Precision: 0.28854    Recall: 0.24800    F1: 0.26674    F2: 0.25517
    Total predictions: 14000    True positives:  496    False positives: 1223    False negatives: 1504    True negatives: 10777

Using the robust feature scaling seems to give little less accuracy and precision, but increases recall. Especially for Gaussian Naive Bayes there is a noteworthy improvement.

#### Next steps (4)

1. Think about where and how to apply PCA
2. Try other algorithms
3. Try different parameter sets
4. Check automatic feature selection
5. Advanced features from text

### 2018-06-03

PCA should be done using a pipeline. Maybe this can include the feature scaling and outlier removal, too. Pipelines can be created using `make_pipeline()` and supplying any order of classifiers as arguments. Pipeline objects can be parameterized dynamically.

First try for using PCA resulted in unchanged prediction quality for Gaussian Naive Bayes, and much worse performance for Linear SVM. But it consumes more time.

Read forums to get an idea if I was using PCA in a wrong way. Instead of hints, I found other peoples reporting about their use of [GridSearchCV for parameter tuning](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) in combination with [cv=StratifiedShuffleSplit()](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) and [RFECV for feature selection](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html). Btw. the CV stands for Cross Validation. Also read about [Gradient-boosted machines](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) which seem to be among the best performing algorithms. Will try those in my solution.

Added a new feature "ratio between total payments and stock value". This boosts performance of especially the SVM (whereas Naive Bayes does not change, not even a bit, so it is not reported again):

LinearSVC(default parameters)
    Accuracy: 0.81136    Precision: 0.32629    Recall: 0.30100    F1: 0.31313    F2: 0.30574
    Total predictions: 14000    True positives:  602    False positives: 1243    False negatives: 1398    True negatives: 10757

Yet again, adding PCA decreases the quality of the results by a factor of 2. Even when playing with the parameters (e.g. set n_components to half the number of features). So I will skip it for now.

Since using PCA in the tester code already hinted that computing times grows to a non-trivial amount, I filled in the code for evaluation within the `poi_id.py` code.s

Added parameters for `--feature-selection` which can be `None`, `SelectKBest`, `RFECV`. First version had some bug or wrong setting, because SVM would not assign _any_ person to the POIs. Need to understand the score which the feature selection outputs for each feature. Does the features with high or low score get selected.

Not sure yet if there is a Wechselwirkung between feature_selection and feature_scaling and if it is good or bad.

The validation code is somewhat unclear. Not sure if I messed it up or if it has been given. Check git history tomorrow!

#### Next steps (5)

1. Review  the last code. Could not bring it to a good state. Finish it.
2. Get feature selection right.

### 2018-06-04

Looking into SelectKBest feature selection, finally understood that the features with the highest score remain. However, the scores are really close to each other and I wonder if there is much difference from the present features.

When printing the three new features, I noticed that they should also be part of the outlier removal.

Played around with Recursive Feature Elimination Cross Validated (RFECV) and found that it gives varying results on each run. The resulting classifier does not label any person as POI and thus the evaluation cannot run. I need to clean up the code again and work more systematically on this.

Also tried SelectPercentile(percentile=68.5) for feature selection. The result is easier to understand and results are quite good for the two classifiers used so far:

Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5)), ('gaussiannb', GaussianNB(priors=None))])
    Accuracy: 0.75057    Precision: 0.25748    Recall: 0.39600    F1: 0.31206    F2: 0.35753
    Total predictions: 14000    True positives:  792    False positives: 2284    False negatives: 1208    True negatives: 9716

Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5)), ('linearsvc', LinearSVC(default_parameters))])
    Accuracy: 0.82107    Precision: 0.24826    Recall: 0.12450    F1: 0.16583    F2: 0.13829
    Total predictions: 14000    True positives:  249    False positives:  754    False negatives: 1751    True negatives: 11246

Results for LinearSVC really depend on the scaling. Seems like normal scaling gives best accuracy and precision but lowest recall, no scaling is on the other side of the scale and robust scaling somewhere between the two. Need to switch to a non-linear kernel anyhow.

Running experiments with `RFECV` against `tester.py` is a PITA. The evaluation in `poi_id.py` is very unstable. Need to change the latter so that it returns fast yet stable results.

Cleaned the code (remove feature selection experiments, make outlier removal part of the pipeline).

Use StratifiedShuffleSplit for more CV-ish evaluation in `poi_id.py` so that I get fast and stable results during development.

#### Next steps (6)

1. Set up GridSearchCV to help with finding optimal algorithm and parameter settings.
2. Get the combination of parameter search, classification and evaluation right.
3. Add some additional relative metrics for the financial data. Actually, most of the features can be set in relation to their total. I am not sure if this is reflected when training a classifier on a multi-dimensional space that contains the feature and the total.
4. Find good algo+params combo.
5. PCA, new text features, ...

### 2018-06-05

Literature search yielded new ideas:

* Try `SelectFromModel(LinearSVC(penalty="l1"))` for feature selection.
* Try other algorithms:
  * `sklearn.ensemble.{RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier}`
  * `sklearn.gaussian_process.GaussianProcessClassifier`
  * `sklearn.linear_model.SGDClassifier`
  * `sklearn.svm.SVC(rbf_kernel)`
  * `sklearn.neural_network.MLPClassifier`

### 2018-06-09

Review and clean up the code. Use code that does not trigger deprecation warnings all over the place. Get the combination of feature selection and the supervised learning algorithm right. Create a baseline.

Baseline:

python poi_id.py --algorithm=decision_tree > blub.txt
python tester.py

Pipeline(memory=None,
     steps=[('decisiontreeclassifier', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'))])
    Accuracy: 0.82053    Precision: 0.31960    Recall: 0.30650    F1: 0.31291    F2: 0.30903
    Total predictions: 15000    True positives:  613    False positives: 1305    False negatives: 1387    True negatives: 11695

Review cross validation and parameter tuning. See below:

#### Review of cross validation

See [Cross validation of estimator performance](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation) for an explanation of StratifiedKFold and StratifiedShuffleSplit and so on. Here is a summary:

* General Idea: Split data into train and test set to avoid overfitting of an estimator.
* Note that CV applies not only to the estimator, but also to preprocessing steps like feature scaling which have to be separately fit and applied on the training data and the test data, respectively, during training and evaluation.
* Simples approach: `train_test_split()`
* When tuning parameters, this will still overfit because knowledge about the test set "leaks" into the model.
* Solution: Split into training, validation and test set: After training, validate on validation set, only finally check against the test set. Note tht parameter tuning is _not_ done against the test but the validation set).
* Problem: Splitting into three sets leaves only a small portion of the data for each step. Thus, neither training generalizes well nor do the test results have much significance. The effect of this problem is larger if the available data is small (as in our project).
* Cross validation (CV) is the solution. Basic k-fold CV splits the training set into k smaller sets ("folds"). Then repeat
  * Use k-1 sets for training
  * Evlatuate against the remaining set
  * Finally the performance metrics of the k steps are averaged.
* Simplest CV approach is the function `cross_val_score()`:

```Python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
>>> Accuracy: 0.98 (+/- 0.03)
```

* Can pass differenct `[score](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)`  functions.
* Parameter `cv` is by default KFold or StratifiedKFold (for ClassifierMixin estimators).
* More elaborate settings with `cross_validate()` (can pass multiple scoring functions at once):

```Python
scores = cross_validate(clf, iris.data, iris.target, scoring=sc['precision_macro', 'recall_macro', 'f1_macro'],
                        cv=5, return_train_score=False)
scores['test_recall_macro'].mean()
```

* Can be used to directly predict with the same interface in function `cross_val_predict()`:

```Python
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics.accuracy_score(iris.target, predicted)
```

* Different "data iterators" for cross validation (only naming the most significant different):
  * `KFold`. See above.
  * `ShuffleSplit`. Shuffle data, split into test and train set. Parameter `n_splits` will repeat this n times and return n different splits (each comprising the whole data).
  * `StratifiedKFold` and `StratifiedShuffleSplit`. Take into account the class labels: The relative class frequency in training and test set corresponds to the whole data set. Useful for small data sets and when there is an imbalance between class labels (as in our project).

#### Review of hyper-parameter tuning

Hyper-parameters are those which are not automatically learnt during the training process. See [scikit's user guid on parameter search](http://scikit-learn.org/stable/modules/grid_search.html#grid-search). Here is a summary:

A search consists of

* an estimator with certain hyper-parameters (e.g. see `SVC().get_params()`)
* a parameter space
* a method for searching or sampling candidates
  * `GridSearchCV` performs search over all combinations of parameters from a grid
  * `RandomizedSearchCV`
* a cross-validation scheme
* a score function

`GridSearchCV` implements the estimator API: After fitting, it can be used as estimator and will use the optimal parameters found.

Some considerations:

* The grid is like a list of parameter sets to be used. There is also something which takes possible values for each parameter and creates the cross-product which can be used as input to the search.
* The default `scoring` function for classification is accuracy. In the project we aim for a good precision and recall, hence these should be used or the f1 score. See the user guide on [scoring functions](http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
  * Specifying multiple scoring functions requires `refit` to be set to one of these. The resulting estimator will use the optimal parameter set for this metric for predictions.
* Evaluation of the resulting model _should_ be done on held-out data, because  searching best parameters is also a kind of training.

Example:

```Python
from sklearn.linear_model import LogisticRegression
param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
                  clf=[SVC(), LogisticRegression()],
                  clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=param_grid)
```

#### Getting the pipeline straight

1. Add new features
2. Remove outliers
3. Select features
4. Scale remaining features
5. Select main algorithm
6. Setup GridSearchCV with StratifiedShuffleSplit (because of small dataset with limited number of positive class labels) as cross validation method.
7. Extract best estimator and save it to disk for later evaluation

#### Adding new features

Created code to generically create features which set an existing feature in relation with some total (e.g. "bonus" / "total payments" => "relative bonus"). Use it to create 11 new features.

Create a feature selection "preview": Compute it on the data extended with new features.

RFECV with StratifiedShuffleSplit as estimation pipeline step slows the execution time of `tester.py` down, because it is nested in the other SSS cycle and thus done a lot of times. Maybe the RFECV can be used as outer estimator to speed it up. Alternatively, its results should be used to restrict the `data_dict` to the relevant features and then store these. This is done by restricting `features_list` to the list of selected feauters.

#### Results

* python poi_id.py --algorithm=gradient_boosting --feature-selection=RFECV && time python tester.py
  * Pipeline(memory=None,
     steps=[('gradientboostingclassifier', GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              m...      presort='auto', random_state=None, subsample=1.0, verbose=0,
              warm_start=False))])
    Accuracy: 0.84593    Precision: 0.44597    Recall: 0.32400    F1: 0.37533    F2: 0.34275
    Total predictions: 14000    True positives:  648    False positives:  805    False negatives: 1352    True negatives: 11195
  * real    1m5.614s

* python poi_id.py --algorithm=ada_boost --feature-selection=RFECV && time python tester.py
  * Pipeline(memory=None,
     steps=[('adaboostclassifier', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=None))])
    Accuracy: 0.84213    Precision: 0.37534    Recall: 0.27700    F1: 0.31876    F2: 0.29232
    Total predictions: 15000    True positives:  554    False positives:  922    False negatives: 1446    True negatives: 12078
  * real    2m47.607s

What I still dont understand: Feature selection gives different features every time I run it, strongly variating in the number and which features are actually selected. However, the result seems to be the same most of the times. I have the random seed of the StratifiedShuffleSplit fixed to 1, so it should always generate the training data for the RFECV. There are extremes:

* Features selected with DecisionTreeClassifier as RFECV estimator. This selects 'poi', 'exercised_stock_options', 'relative_bonus'.
  * Accuracy: 0.80992    Precision: 0.36489    Recall: 0.31800    F1: 0.33983    F2: 0.32639
  * When repeated, this gives really different sets of selected features everytime.
* No features filtered away: python poi_id.py --algorithm=gradient_boosting && time python tester.py
  * Accuracy: 0.84787    Precision: 0.36847    Recall: 0.19750    F1: 0.25716    F2: 0.21770

If I don't find the reason by the evening, I will post the question to the forums.

* Decision tree has a `random_state` parameter, too. Here it is really at will which feature is selected for splitting. Set the `criterion="entropy"` instead of default `"gini"`.

#### More algorithms

Trying  `SelectFromModel(LinearSVC(penalty="l1"))` for feature selection.

* Does not improve the results (for GradientBoostingClassifier).

Trying other algorithms:

* `sklearn.linear_model.LogisticRegression` (+ SelectFromModel(LinearSVC), fails on --feature-selection=RFECV)
  * Accuracy: 0.87373  Precision: 0.56709  Recall: 0.22400  F1: 0.32115  F2: 0.25484
  * Really fast
* `sklearn.ensemble.RandomForestClassifier` (+ RFECV)
  * Accuracy: 0.86087  Precision: 0.42461  Recall: 0.12250  F1: 0.19014  F2: 0.14282
* `sklearn.gaussian_process.GaussianProcessClassifier` (+ SelectFromModel(LinearSVC), fails on --feature-selection=RFECV)
  * Accuracy: 0.88093  Precision: 1.00000  Recall: 0.10700  F1: 0.19332  F2: 0.13027
* `sklearn.linear_model.SGDClassifier`
  * Accuracy: 0.79053  Precision: 0.23662  Recall: 0.25650  F1: 0.24616  F2: 0.25226 (--feature-selection=linear_model)
  * Accuracy: 0.77913  Precision: 0.03800  Recall: 0.02700  F1: 0.03157  F2: 0.02866 (--feature-selection=RFECV, leaves only a few features)
* `sklearn.svm.SVC(rbf_kernel)`
  * Fails in evaluation in `tester.py`.
* `sklearn.neural_network.MLPClassifier` (+ SelectFromModel(LinearSVC), fails on --feature-selection=RFECV)
  * Accuracy: 0.73647  Precision: 0.13712  Recall: 0.18450  F1: 0.15732  F2: 0.17258

Summary: `GradientBoostingClassifier` and `AdaBoostClassifier` are best, allthough slow. The former is much faster than the latter.

* `ensemble.GradientBoostingClassifier`
  * Accuracy: 0.81900  Precision: 0.20816  Recall: 0.12750  F1: 0.15814  F2: 0.13821 (--feature-selection=linear_model)
  * Accuracy: 0.85087  Precision: 0.38595  Recall: 0.20050  F1: 0.26390  F2: 0.22182(--feature-selection=RFECV)
  * Accuracy: 0.84780  Precision: 0.36862  Recall: 0.19850  F1: 0.25804  F2: 0.21868 (no feature selection)
* `ensemble.AdaBoostClassifier`
  * Accuracy: 0.85867  Precision: 0.46073  Recall: 0.35200  F1: 0.39909  F2: 0.36944 (--feature-selection=RFECV, features: ['poi', 'salary', 'bonus', 'other', 'expenses', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'from_poi_to_this_person', 'emails_to_poi_share', 'total_payments_to_stock_value_ratio', 'relative_long_term_incentive', 'relative_deferred_income', 'relative_other', 'relative_exercised_stock_options', 'relative_restricted_stock'])
  * Accuracy: 0.79433  Precision: 0.15686  Recall: 0.12400  F1: 0.13851  F2: 0.12942 (--feature-selection=linear_model, features: ['poi', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'relative_salary', 'relative_deferred_income', 'relative_other', 'relative_exercised_stock_options'])

#### Hyper-parameter tuning

Algorithm `GradientBoostingClassifier` seems to work best/fast with default settings. Might be luck, but let's see if we can improve this one by tuning its parameters.

* First attempt. Best parameters: {'gradientboostingclassifier__max_features': 'sqrt', 'gradientboostingclassifier__max_depth': 3, 'gradientboostingclassifier__subsample': 0.8, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__loss': 'exponential'}
  * Accuracy: 0.87693  Precision: 0.60294  Recall: 0.22550  F1: 0.32824  F2: 0.25777
* Without "THE TRAVEL AGENCY IN THE PARK". {'gradientboostingclassifier__max_features': None, 'gradientboostingclassifier__max_depth': 9, 'gradientboostingclassifier__subsample': 1.0, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__loss': 'deviance'}
  * Accuracy: 0.82620  Precision: 0.32707  Recall: 0.28700  F1: 0.30573  F2: 0.29421
* Select parameters with scoring function "recall". {'gradientboostingclassifier__max_features': None, 'gradientboostingclassifier__max_depth': 9, 'gradientboostingclassifier__subsample': 1.0, 'gradientboostingclassifier__n_estimators': 100, 'gradientboostingclassifier__loss': 'deviance'}
  * Accuracy: 0.82620  Precision: 0.32707  Recall: 0.28700  F1: 0.30573  F2: 0.29421
* {'gradientboostingclassifier__criterion': 'friedman_mse', 'gradientboostingclassifier__max_depth': 8, 'gradientboostingclassifier__n_estimators': 50, 'gradientboostingclassifier__max_features': None, 'gradientboostingclassifier__subsample': 1.0, 'gradientboostingclassifier__loss': 'deviance'}
  * Accuracy: 0.83427  Precision: 0.35518  Recall: 0.29800  F1: 0.32409  F2: 0.30791

Recall still bad. Removing the travel agency does not seem to help, but it should be gone anyhow.

#### Next steps (7)

1. Reflect on the whole thing.
2. Read on RFECV and GradientBoostingClassifier.
3. Review the new features: Should missing data points be NaN or 0? Maybe I changed this in the `featureFormat()` arguments. Visualize them.
4. Try feature scaling and outlier removal.
5. Try using RFECV as estimator like done by that other dude.

No findings on the NaN values. Changed the calls to `featureFormat()` back so that NaN are removed (except for outlier detection where we need a special structure of the data).

Somehow the GridSearchCV became much slower, allthough I did not change anything that might influence the performance.

* GradientBoostingClassifier, feature selection RFECV mit DecistionTreeClassifier, outlier removal after new features (selects just two features).
  * Accuracy: 0.79791  Precision: 0.42894  Recall: 0.33650  F1: 0.37714  F2: 0.35166
* GradientBoostingClassifier, feature selection RFECV mit DecistionTreeClassifier, outlier removal after new features (selects just two features).{'gradientboostingclassifier__max_features': None, 'gradientboostingclassifier__max_depth': 8, 'gradientboostingclassifier__n_estimators': 40, 'gradientboostingclassifier__criterion': 'friedman_mse', 'gradientboostingclassifier__loss': 'deviance'}
  * Accuracy: 0.81609  Precision: 0.49463  Recall: 0.53000  F1: 0.51171  F2: 0.52253
* Same, but with p68.5 instead of RFECV feature selection and parameter choosing.
  * Accuracy: 0.81964  Precision: 0.33354  Recall: 0.26300  F1: 0.29410  F2: 0.27462

### 2018-06-10

Tried the setup from the forum with a variation of my preprocessing steps:

```sh
python poi_id.py --algorithm=gradient_boosting  --remove-outliers && time python tester.py
  [...]
  RFECV(cv=StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.3,
            train_size=None),
   estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=7,
              max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=15, min_samples_split=75,
              min_weight_fraction_leaf=0.0, n_estimators=75,
              presort='auto', random_state=None, subsample=0.8, verbose=0,
              warm_start=False),
   n_jobs=1, scoring=None, step=1, verbose=0)
    Accuracy: 0.84707    Precision: 0.39648    Recall: 0.13500    F1: 0.20142    F2: 0.15551

python poi_id.py --algorithm=gradient_boosting && time python tester.py
  [...]
  RFECV(cv=StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.3,
            train_size=None),
   estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=7,
              max_features='sqrt', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=15, min_samples_split=75,
              min_weight_fraction_leaf=0.0, n_estimators=75,
              presort='auto', random_state=None, subsample=0.8, verbose=0,
              warm_start=False),
   n_jobs=1, scoring=None, step=1, verbose=0)
    Accuracy: 0.86007    Precision: 0.43038    Recall: 0.15300    F1: 0.22575    F2: 0.17564

```

Performance not really better.

#### Next steps (8)

* Use feature "email_address" if possible, or convert to numerical "has_enron_email_address" in {0, 1}.
* Review my first experiments. Why was the performance so much better? Was it just the usage of different `train_test_split()`?

### 2018-06-12

Added email feature. Consider Kenneth Lay as outlier and drop him from the data.

Try with and without additional outlier removal. Try with robust feature scaling:

```sh
python poi_id.py --algorithm=gradient_boosting --feature-selection=p68.5 --remove-outliers && time python tester.py
  Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5,
         score_func=<function f_classif at 0x7fc5649c98c0>)), ('gradientboostingclassifier', GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_...         presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False))])
    Accuracy: 0.84093    Precision: 0.38684    Recall: 0.19400    F1: 0.25841    F2: 0.21548

python poi_id.py --algorithm=gradient_boosting --feature-selection=p68.5  && time python tester.py
Pipeline(memory=None,
     steps=[('selectpercentile', SelectPercentile(percentile=68.5,
         score_func=<function f_classif at 0x7fbec21728c0>)), ('gradientboostingclassifier', GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_...         presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False))])
    Accuracy: 0.84840    Precision: 0.36355    Recall: 0.18250    F1: 0.24301    F2: 0.20269

python poi_id.py --algorithm=gradient_boosting --feature-selection=p68.5 --feature-scaling=robust && time python tester.py
  Pipeline(memory=None,
     steps=[('robustscaler', RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)), ('selectpercentile', SelectPercentile(percentile=68.5,
         score_func=<function f_classif at 0x7fc9e90db8c0>)), ('gradientboostingclassifier', GradientBoostingClassifier...         presort='auto', random_state=1, subsample=1.0, verbose=0,
              warm_start=False))])
    Accuracy: 0.84847    Precision: 0.36418    Recall: 0.18300    F1: 0.24359    F2: 0.20322
```

Feature scaling seems to make almost no difference for GradientBoosting.

DecisionTreeClassifier with all features comes quite close to the require .3 precision and recall. Could be fine-tuned to reach it.

AdaBoost is around .2, not very good. Probably needs more tuning.

LinearSVM reached the requirements:

```sh
python poi_id.py --algorithm=linear_svc --feature-scaling=normal && time python tester.py
  Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linearsvc', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=1, tol=0.0001,
     verbose=0))])
    Accuracy: 0.83140    Precision: 0.35710    Recall: 0.33050    F1: 0.34329    F2: 0.33550

python poi_id.py --algorithm=linear_svc --feature-scaling=robust && time python tester.py
  Pipeline(memory=None,
     steps=[('robustscaler', RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)), ('linearsvc', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=1, tol=0.0001,
     verbose=0))])
    Accuracy: 0.80973    Precision: 0.24704    Recall: 0.20850    F1: 0.22614    F2: 0.21521

python poi_id.py --algorithm=linear_svc --feature-scaling=normal --remove-outliers && time python tester.py
  Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linearsvc', LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=1, tol=0.0001,
     verbose=0))])
    Accuracy: 0.82229    Precision: 0.37776    Recall: 0.37700    F1: 0.37738    F2: 0.37715

Checked some more variations, none performs similar.
```

The SVM with linear kernel, linear feature scaling and removal of additional 5% outliers and using all available features seems to work best. Will stop to search for alternatives, this is good enough and the search, on the other hand, will consume an arbitrary amount of time.

#### Next steps (9)

1. Finalize the code: Make grid search controllable by parameter.
2. Do some parameter tuning for the best values on SVC.
3. Unleash the random seed.
4. Clean code and submit.
5. Think about why SVM with RBF kernel does not work here.

### 2018-06-13

Set default parameters so that someone calling just `python poi_id.py` will get the my selected algorithm and parameter combination.

Unleashed the random seed, results do not vary for my choices.

Added parameter for enabling GridSearch. Added new module for returning predefined parameter grids as well as the best found parameter sets.

#### Final steps

1. One last try: Recursive feature elimination with linear SVM + parameters.
2. Write the `report.md`.
3. Review code documentation and style. Just get the basics straight.

### 2018-06-14

Created scaffold for the report. Added MinMax-scaler.

While giving RFE a last try, found that it also handles the `scoring` parameter. Passing the same function `"f1"` now as for `GridSearchCV`. This, for the first time, yields a reasonable subset of features and has > .3 precision and recall:

```sh
python poi_id.py --feature-selection=RFECV && time python tester.py

Selected features: ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'other', 'director_fees', 'expenses', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'total_payments_to_stock_value_ratio', 'has_enron_email_address', 'relative_restricted_stock']

Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linearsvc', LinearSVC(C=200, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=None, tol=0.0001, verbose=0))])
    Accuracy: 0.81220    Precision: 0.32107    Recall: 0.36650    F1: 0.34228    F2: 0.35641
```

However, the next run had really bad results again. Maybe the CV setting needs to be separate for this one. Might need many more folds to find a stable set of features.

Increased `n_splits` to 100, did GridSearchCV again:

```sh
python poi_id.py --feature-selection=RFECV --perform-parameter-search && time python tester.py

Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linearsvc', LinearSVC(C=20, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0))])
    Accuracy: 0.81627    Precision: 0.35257    Recall: 0.45200    F1: 0.39614    F2: 0.42787
```

Finally tried PCA again, still not improvements.

Wrote most of the report.

### 2018-06-15

Added description of the task.

Added output of feature importances and ranks during feature selection. This shows that RFECV with GradientBoostingClassifier returns a `_get_support_mask()` which discards features with high importance. Is this a bug? Prepared a forum question. Will review it tomorrow after handing in the first solution.

Described tuned parameters.

Explained metrics.

Created literature listing.

Submitted.

### 2018-06-{16,17,19,20}

Worked on extracting text features from email. Stopped after I found that there is not enough email data for persons in the data set while I was mapping person names to maildirs. Documented the findings.

Extended report about the importance of hyper-parameter tuning. Clarified certain other aspects. Resubmitted.