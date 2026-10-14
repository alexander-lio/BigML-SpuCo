# Spurious Correlations on SpuCoMNIST: ERM, then GEORGE-style group balancing

Notebook (`spuco_george_asg.ipynb`) implementing the three-step GEORGE pipeline (ERM, class-wise clustering of ERM features, group-balanced retraining) with the [SpuCo](https://spuco.readthedocs.io/en/latest/) package, following the [SpuCo quickstarts](https://github.com/BigML-CS-UCLA/SpuCo/tree/master/quickstart/spuco_mnist). By Alexander Lio (alio at ucla dot edu).


All results come from a single run with seed 42, evaluation done on a held-out test set. Results shown in notebook and outputs.

## Discussion


**ERM.** At 0.995, ERM is 100% on the five groups where color matches the class and 0% on the other 20, so it predicts from background color.

**Group balancing helps in relation to group inference.** At 0.995 there are just 243 minority images. The clustering isolated all minority examples for classes 0, 2 and 4, and none for classes 1 and 3; the first three average about 65% off-diagonal and the last two about 16%. The three groups still at 0% are all in classes 1 and 3.

**With more minority data the same pipeline produces better results** At 0.9, about 10% of the training set is minority, and every test group is between 88% and 99%, at a cost of about 2 points on the weighted average. This suggests the weak 0.995 result came from lower amounts of minority data and not from the method, though I tested it with only one seed and one changed setting.

**Limitations.** Single seed, no hyperparameter search, and `GeorgeInference` returned many more groups than `max_clusters=5` (18 at 0.995), so the majority groups were split. I did not check how `GroupBalanceBatchERM` samples groups, so I can't say how that split affects each group's share of a batch.

**Assumptions.**  Multiple seeds; vary the clustering settings and compare with plain per-class KMeans on the same features could be beneficial for the experiment.

## Files

- `spuco_george_asg.ipynb`: the full pipeline with saved outputs.
- `outputs.pdf`: Just outputs of notebook code