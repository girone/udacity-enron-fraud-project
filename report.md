# Project Report: Identify Fraud from Enron Email

This project evolves around the Enron scandal. The goal is to train a classifier on financial and email data and to use it to detect persons involved in the fraud case.

## Data exploration

The present data set contains 146 data points. 18 of these are persons of interest (POI). For each element, the data set lists 20 features plus the class label. However, many features are unknown for several persons. During the investigation, these will be handled as NaN or 0.

There are several outliers in the data. The most obvious is the entry "TOTAL" which is merely the row of sums for each feature from the original PDF. Then there is "THE TRAVEL AGENCY IN THE PARK" which has just one feature entry for "Other". As it does not correspond to a single person which could possibly be a POI, it is considered as outlier, too. Finally, there is the entry for Kenneth Lay, the key figure in the Enron scheme who, during his time as CEO, earned so much that his numbers would bias the data set when left inside. Other than these obvious outliers, I added the optional command linee argument `--remove-outliers`. When enabled, the nearest-neighbor based method `LocalOutlierFactor` is employed to detect and remove the 5% data points which are most likely outliers.

## Feature selection

In addition to the 20 baseline features, I added up to 16 new features. For that purpose, I created a new module `additional_features` from which generators for the new features can be included. The new features are

* 12 relative financial features
  * Each of the financial data features is set in relation with the sum of all features of the same category. E.g., "relative_bonus" is "bonus" divided by "total_payments".
* One feature for the relation between "total_payments" and "total_stock_value".
* One feature about the email address
  * Does a person have an Enron email address or not? Takes values 1 or 0, respectively.
* Two features for the share of emails which are addressed to a POI or received from a POI for each person, respectively. Note that these features are based on the features "from_this_person_to_poi" and "from_poi_to_this_person", which should be considered invalid for this classification task. This is because they are computed from the known set of POIs. However, when training and evaluating our classificators, we do not know who is a POI in the test set. Thus, the feature should really be computed on the fly, considering only POIs in the training set.

My expectation was that relative features better capture patterns than absolute features, in the sense that when of two otherwise comparable persons one writes less emails, it will write less emails to a POI. Setting the numbers in relation accounts for persons writing more or less emails, earning more or less, and so forth.

The email address feature maybe gives a clue if a person is actually an Enron employee or not.

Some algorithms benefit from scaled features, for example SVMs. I added `--feature-scaling` to `poi_id.py` which can select between normal Gaussian scaling with mean and stddev, outlier robust scaling and no scaling at all. It turned out that for algorithms which do require scaled features, `--feature-scaling=normal` gave the best results.

I tested the algorithms with different subsets of these 36 features. I added the argument `--feature-selection` to the script. It allows to switch between all features, selection of the k best, the best 68.5%, feature selection by `LinearModel(LinearSVC())` or feature selection by recursively eliminating irrelevant features using cross validation (`RFECV`). The latter will use the selected algorithm itself as internal estimator, or if that is not applicable, a `DecisionTreeClassifier` and a scoring function that corresponded to the one used when training the algorithm, which was typically `"f1"`. The best results have been achieved by `--feature-selection=RFECV` or no feature selection at all. Here is an examplary output of the feature selection:

|rank|importance|feature|selected?|
|----|----------|-------|---------|
| 1.| 23.389828822 | relative_bonus | selected |
| 2.| 16.8003517636 | exercised_stock_options | selected |
| 3.| 16.3125076135 | total_stock_value | selected |
| 4.| 15.282789091 | relative_long_term_incentive | selected |
| 5.| 14.0800023887 | emails_to_poi_share | selected |
| 6.| 13.9542854043 | bonus | selected |
| 7.| 12.3815615903 | salary | selected |
| 8.| 11.8497813806 | deferred_income | selected |
| 9.| 7.08626103486 | shared_receipt_with_poi | selected |
| 10.| 5.94789081886 | has_enron_email_address | selected |
| 11.| 5.23275955039 | long_term_incentive | selected |
| 12.| 4.97016273388 | expenses | selected |
| 13.| 4.42830698268 | from_poi_to_this_person | selected |
| 14.| 3.48810567249 | relative_salary | selected |
| 15.| 3.31485543549 | restricted_stock | selected |
| 16.| 3.21442216665 | emails_from_poi_share | selected |
| 17.| 2.61649394958 | from_this_person_to_poi | selected |
| 18.| 1.9903678897 | director_fees | selected |
| 19.| 1.75226840787 | total_payments | selected |
| 20.| 1.31240379678 | relative_restricted_stock | selected |
| 21.| 1.1988587529 | relative_deferral_payments | selected |
| 22.| 1.15861419331 | relative_other | selected |
| 23.| 1.05638708178 | to_messages | selected |
| 24.| 0.716307922805 | relative_restricted_stock_deferred | selected |
| 25.| 0.228156379744 | relative_loan_advances | discarded |
| 26.| 0.221188632191 | deferral_payments | discarded |
| 27.| 0.210615297495 | total_payments_to_stock_value_ratio | discarded |
| 28.| 0.194927211265 | relative_director_fees | discarded |
| 29.| 0.186247086247 | loan_advances | discarded |
| 30.| 0.137096503183 | relative_deferred_income | discarded |
| 31.| 0.131732481639 | from_messages | discarded |
| 32.| 0.0801979619203 | relative_exercised_stock_options | discarded |
| 33.| 0.0613680272636 | restricted_stock_deferred | discarded |
| 34.| 0.00911576467536 | other | discarded |
| 35.| 0.00113575916967 | relative_expenses | discarded |

Principal component analysis PCA can help to reduce the dimensionality of classification problems, which speeds up training and prediction times of classificators. For some classification algorithms, it also improves the quality of the results. The `poi_id.py` has the option `--perform-PCA` which will transform the selected features and keep only the 75% with highest variance. However, in the present task I could not observe any improvements that would justify the increased preprocessing time.

I would have liked to further the parameters by more elaborate email features. For example, finding the words with highest tf.idf scores among all emails and compare the top 100 words between persons sounds very promising. Another approach would be to look for words with high tf.idf from POIs and then take each word's score directly as new feature. To do that, I would need to come up with a metric to compare these sets of words and incorporate them into the otherwise univariate features. Unfortunately this goes beyond my time frame.

## Algorithms

I investigated a multitude of algorithms (see `python poi_id.py --help` on `--algorithm` for a full list) to solve the present task. I did not have the time to tune parameters for each of them, so I discontinued investigating those for which I could not guess good initial parameters within a few tries.

A noteworthy option is `rbf_svc` which employs a Support Vector Machine with a radial basis function, because I could not find a parameter setting which would not result in a classifier that does not discard all POIs.

For the most promising algorithms I performed a search for the best parameters using `GridSearchCV` and the parameter grids in the module `custom_param_grids`.

For `LinearSVC` the only parameter to tune is the `C` , the penalty for error terms. I tried values `[0.1, 0.5, 1.0, 10, 100, 200]`. The other parameters conflict with each other (`loss="hinge"` does not work when `penalty="l1"` or `dual=False` and so on). The best parameter found are:

```Python
    "linearsvc": {
        "C": 200,
        "fit_intercept": True,
        "max_iter": 1000,
        "penalty": "l2",
        "class_weight": None,
        "multi_class": "ovr",
        "dual": True,
        "verbose": 0,
        "tol": 0.0001,
        "intercept_scaling": 1,
        "random_state": None,
        "loss": "hinge"
    }
```

`GradientBoostingClassifier` is different in that it has many more parameters that can be tuned. See `custom_param_grids.py` for the different values for the individual parameters I tested. The search is time exausting because there are so many combinations and each run takes 1-2 minutes. The best setting I found is:

```Python
    "gradientboostingclassifier": {
        "criterion": "friedman_mse",
        "max_depth": 8,
        "n_estimators": 100,
        "max_features": None,
        "subsample": 1.0,
        "loss": "deviance"
    }
```

I stored the best settings in the same module, so that they are loaded automatically when running `poi_id.py` without any parameters (except for `--algorithm`).

## Evaluation

I discarded the original evaluation and always checked the results against `tester.py` because I found that the results to the former did not generalize well. This is because the size of the data set is limited, and the POIs only are a small percentage thereof.

This condition makes the use of cross-validation mandatory: We need to avoid overfitting and want to get a result that generalizes well. Instead of splitting the data one time into a training and a test set, this is repeated several times with different, randomly sampled splits. Thereby the relative frequency of POIs is approximately the same in the test and training set as in the complete data set, which is important if there are just a few such samples in the data. This has been achieved using `sklearn.model_selection.StratifiedShuffleSplit()`. The resulting metrics for accuracy (rate of true predicted labels against all predictions), precision (share of correctly identified POIs and total predicted POIs), recall (share of identified POIs to total POIs in the data set) and f1 (geometric mean of precision and recall) are the average of the results for the individual splits.

Besides of evaluating the metrics I did some functional tests for scaling (see `custom_validation.py`).

## Results

|algorithm|accuracy|precision|recall|f1|duration (1.)|
|--|--|--|--|--|--|
| LinearSVC (2.a) | 0.81033 | 0.35181 | 0.50150 | 0.41352 | 15s |
| LinearSVC (2.b) | 0.80467 | 0.35257 | 0.45200 | 0.39614 | 15s |
| GradientBoostingClassifier (3.a) | 0.84353 | 0.38942 | 0.30550 | 0.34239| 90s |

1. Approximate time for preprocessing, training and prediction measured when running on both cores of a 2x2.53GHz notebook from 2008. Does not include time for best parameter search. Just to get an idea.
2. Parameters for SVC: --scaling=normal, kernel="linear"
  a) C=200, dual=True, loss="hinge", penalty="l2"
  b) C=20, dual=True, loss="squared_hinge", penalty="l2"
3. Parameters for GradientBoostingClassifier:
  a) criterion="friedman_mse", max_depth=8, n_estimators=100, max_features=None, subsample=1.0, loss="deviance"

The best solution has been found with `LinearSVC`. It has an accuracy of ~81% which means it assigns 81 out of 100 data points to the correct class. The precision is 35%, meaning that it out of all persons labelled as POI, 35% are actually POIs. The recall is 50%, which means that of all actual POIs the classifier labels every second correctly as POI.