# Titanic Passenger Classification — Lab 5

## EDA findings

- The supplied teaching sample contains **100 rows** and five columns: `PClass`, `Gender`, `Sibling`, `Embarked`, and `Survived`.
- There are **0 missing values** and **56 exact duplicate rows**.
- The target is balanced: **50 survived** and **50 did not survive**. The majority-class accuracy baseline is **50%**.
- Passenger-class counts: Third 60, Second 20, First 20.
- Gender counts: Male 58, Female 42.
- Sibling counts: Zero 59, One 29, Two 6, Three 6.
- Embarkation counts: Southampton 75, Cherbourg 16, Queenstown 9.
- Because duplicate rows may occur across train/test and CV folds, these scores can be optimistic. Deduplicate or group records before relying on the estimates.

## Three-model comparison

The holdout split is stratified (80/20, random state 42). The extended feature set adds `FamilySize` and `IsAlone`. Five-fold stratified cross-validation uses the extended features.

| Model | Base holdout accuracy | Extended holdout accuracy | Delta | 5-fold CV mean ± SD (extended) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.500 | 0.500 | +0.000 | 0.700 ± 0.114 |
| Decision Tree | 0.550 | 0.550 | +0.000 | 0.680 ± 0.112 |
| Random Forest | 0.500 | 0.500 | +0.000 | 0.640 ± 0.124 |

## Tuning and evaluation

- Random Forest grid-search parameters: `max_depth=3`, `min_samples_split=2`, `n_estimators=100`.
- Best five-fold cross-validated accuracy: **0.7000**.
- Tuned-model holdout accuracy: **0.5000**; ROC-AUC: **0.6150**.
- Confusion matrix (true rows [No, Yes], predicted columns [No, Yes]): `[[6, 4], [6, 4]]`.
- The holdout metrics are measured before refitting the final deployment pipeline on all supplied rows.
- ROC curve: [figures/roc_curve.png](figures/roc_curve.png).

## Deployment recommendation

Keep `titanic_best_model.joblib` as the coursework deployment artifact, but do not treat its current performance as sufficient for real-world use: the held-out accuracy equals the 50% majority baseline, the sample is tiny and duplicated, and the ROC-AUC is only 0.615. The saved fitted pipeline bundles its categorical encoding and accepts the four categorical fields plus derived `FamilySize` and `IsAlone` fields. The separate `*_encoder.pkl` files are provided for Laboratory 6 compatibility; preserve the feature schema and validate incoming categories. Obtain a larger, representative dataset and evaluate on de-duplicated, unseen passengers before making any deployment claim.

## Reproducibility

Run [titanic_classification.ipynb](titanic_classification.ipynb) from this folder. It reads the included `sample-data.csv` and regenerates the model, encoders, ROC plot, and this report. Imputation and one-hot encoding are fitted inside each training fold.
