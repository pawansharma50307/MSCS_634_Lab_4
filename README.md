# MSCS 634 Lab 4: Regression Analysis with Regularization Techniques

**Name:** Pawan Sharma
**Course:** Advanced Big Data and Data Mining (MSCS-634-M50)

## Purpose

In this lab I used the Diabetes dataset from `sklearn.datasets` (442 patients, 10 baseline health measurements) to predict disease progression one year later. I built simple linear, multiple, and polynomial regression models, then added Ridge and Lasso regularization to see how they handle overfitting. Every model was trained and tested on the same 80/20 split and scored with MAE, MSE, RMSE, and R² so the results could be compared directly.

## Files

- `Lab4_Regression_Analysis.ipynb` – the full notebook with code, outputs, charts, and discussion
- `README.md` – this summary

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Lasso, degree-2 polynomial (α = 2.42) | 40.83 | 50.46 | **0.519** |
| Ridge, degree-2 polynomial (α = 28.9) | 40.04 | 50.91 | 0.511 |
| Lasso, 10 features (α = 2.42) | 42.91 | 52.90 | 0.472 |
| Ridge, 10 features (α = 58.8) | 43.07 | 53.43 | 0.461 |
| Multiple linear regression | 42.79 | 53.85 | 0.453 |
| Polynomial degree 2, no regularization | 43.58 | 55.64 | 0.416 |
| Polynomial degree 6, BMI only | 51.12 | 63.08 | 0.249 |
| Simple linear regression, BMI only | 52.26 | 63.73 | 0.233 |

## Key Insights

- **One feature isn't enough.** BMI is the strongest single predictor, but on its own it only explained about 23% of the variance. Using all 10 features nearly doubled R² to 0.45.
- **Polynomial terms overfit without regularization.** Expanding to 65 degree-2 terms raised the training R² to 0.61 but dropped the test R² to 0.42, worse than plain multiple regression. On BMI alone, going past degree 7 made test error explode and R² went negative.
- **Regularization fixed the overfitting and produced the best models.** Adding a Lasso penalty to the same 65 polynomial features raised the test R² to 0.52, the best score in the lab. Lasso kept 22 of the 65 terms and set the other 43 to zero.
- **Ridge handled multicollinearity.** `s1` and `s2` are about 90% correlated, and ordinary least squares gave them large, opposite-sign coefficients that cancelled each other out. Ridge shrank them to small, stable values.
- **Alpha controls the bias–variance trade-off.** Very small alpha behaves like ordinary regression (overfits), a moderate alpha gives the best test score, and a large alpha shrinks everything toward zero (underfits). Lasso collapses faster because it drops features completely.
- **About the data:** BMI, `s5`, and blood pressure were the most important predictors in every model. Even the best model only explains about half the variation in disease progression, which says more about the limits of 10 baseline measurements than about the models.

## Challenges and Decisions

- **Choosing the single feature:** I used BMI for the simple linear model because it had the highest correlation (0.59) with the target.
- **Scaling:** sklearn already scales these features, but squared and interaction terms of small numbers become even smaller. I put a `StandardScaler` inside each polynomial and regularized pipeline so the penalty treats every term fairly.
- **Picking alpha:** I first swept 40 alpha values to show how alpha changes the results, which meant looking at test scores. To make sure I wasn't just tuning to the test set, I also ran `RidgeCV` and `LassoCV` with 5-fold cross-validation on the training data only. They chose somewhat different alphas but gave very similar test scores (0.51–0.52 for the polynomial models), so the improvement holds up.
- **Showing overfitting clearly:** With all 10 features, high polynomial degrees create hundreds of terms and are hard to visualize, so I used BMI alone for the degree experiment (degrees 1–15), where the overfit curve is easy to see.
- **Small dataset:** With only 442 rows, results move a bit depending on the split, so I treat R² differences of about 0.01 between close models as noise.

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Lab4_Regression_Analysis.ipynb
```
