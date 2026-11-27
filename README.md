# pipeguard-ml

Pipeguard is a zero-runtime-dependency AST linter for preprocessing data leakage in scikit-learn and pandas workflows. It scans Python scripts and Jupyter notebooks without importing their dependencies or executing their code.

## Install and run

Requires Python 3.10 or later. Install from this checkout:

```sh
python -m pip install .
pipeguard train.py notebooks/
```

From the checkout you can also run `python -m pipeguard train.py` without installing. Directory inputs are scanned recursively. The CLI exits with 0 for a clean scan, 1 for findings, and 2 for input or parse errors. Findings include the file, line, column, and rule code. Notebook findings also include the one-based cell number, counting Markdown cells.

The distribution name is `pipeguard-ml`. The installed command is `pipeguard`.

## Rules

| Code | Reports |
| --- | --- |
| PG001 | A supported scikit-learn preprocessor learns from data that flows into a later recognized split. This covers `fit`, `fit_transform`, `partial_fit`, and helpers such as `scale`. |
| PG002 | A supported scikit-learn preprocessor learns from data derived from the test output of a recognized split. |
| PG003 | A statistic such as `mean`, `median`, or `std` flows into the input of a recognized split, as in pandas imputation or standardization. |
| PG004 | Features learned by a supported scikit-learn preprocessor reach a cross-validation or parameter-search call. |

For example, both preprocessing calls below use rows that can end up in the test set:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.fillna(df.median())                  # PG003 at median()
X = StandardScaler().fit_transform(X)      # PG001
X_train, X_test = train_test_split(X, random_state=42)
```

Split first, then fit the imputer and scaler on the training partition:

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test = train_test_split(df, random_state=42)
preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.transform(X_test)
```

## Cross-validation

Fitting a scaler on the training partition before cross-validation still exposes each validation fold to statistics from its own rows. Pipeguard reports PG004 at the preprocessing line when the transformed features reach a supported evaluation call.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scaled = StandardScaler().fit_transform(X_train)  # PG004
scores = cross_val_score(model, scaled, y_train)

pipeline = make_pipeline(StandardScaler(), model)
scores = cross_val_score(pipeline, X_train, y_train)  # scaler fitted inside each fold
```

The rule recognizes `cross_val_score`, `cross_validate`, `cross_val_predict`, and calls to `.fit()` on `GridSearchCV` and `RandomizedSearchCV`. It tracks scikit-learn preprocessing outputs passed as features. Hand-written fold loops and pandas statistics before cross-validation are outside this rule's scope.

## Grouped holdout example

The Kepler Exoplanet Transit Classifier holds out entire stars using a project helper. Run this from its checkout:

```sh
pipeguard --index-split src.splitting.group_split src/train.py
```

Use `--index-split` for an imported helper that takes features as its first positional argument or as `X=` and returns `(train_indices, test_indices)`. The original Kepler training script scanned clean. Adding a full-data scaler fit immediately before `group_split` in a temporary copy produced PG001. Fitting the scaler on `X_train_raw` after the split cleared the finding. [Kepler training script](https://github.com/Shadowsama20/Exoplanet-Transit-Classifier/blob/main/src/train.py)

## Analysis scope

Pipeguard recognizes explicit scikit-learn imports, common preprocessing modules, simple aliases, `train_test_split`, and configured index-split helpers. It analyzes notebooks in saved cell order and skips common IPython line magics and non-Python cell magics. Its checks use local data flow and source order. Dynamic calls, unconfigured custom splitters, general data flow across helper calls, and notebook execution history are outside its model. The pandas rule matches method names and can also report statistics from pandas-like objects. Review findings in context.
