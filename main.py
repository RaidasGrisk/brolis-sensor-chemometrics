'''
https://www.youtube.com/watch?v=CEUa1JgKcp0&ab_channel=VernierScienceEducation

Noteworthy observations:

    ▪ Instead of scaled raw absorbance data at each wavelength, do % change in between subsequent wavelengths.
    This completely does not work and I've no understanding why. Does absorbance intercept manner that much?
    spectra_ = spectra_.pct_change(axis=1).drop(0, axis=1)

    ▪ Plot absorbance (y) vs concentration (x) at each wavelength. Expect it ta have a ~linear relationship?
    Why there's no visible relationship..?

    ▪ Using ridge regression (regularize by keeping the squares of coefficients as low as possible), we can see which
    wavelengths are most used by the model to predict y. Some ranges are more useful than other. e.g. ~2110 and ~2300.
    See ridge_feature_importance.py for visualization.

    ▪ Currently, the only way to come to reasonable results is to either use PCA or regularize heavily to reduce X.
    RandomForest, while not the best algo for this, could be used to pick features. Not now as not enough data points.

    ▪ Too little data for RNN? No. There's no signal in the sequence of absorptivity across different wavelengths.
    The signal lies in specific wavelengths, not the sequence.

    ▪ How do I interpret PCA that is done across the wavelengths? Specific PCA components shows very high (~70 to 95%)
    correlation with target concentration.

TODO: try PCA with polynomials?

Best models:
PCA(5) + Poly(2) + Ridge    - 0.924
PCA(80) + CNN               - 0.922

'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import feature_selection
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# fix black pylab window issue
# https://youtrack.jetbrains.com/issue/PY-52360/Pycharm-opens-an-unresponsive-black-window-when-trying-to-plot
import matplotlib
matplotlib.use('Qt5Agg')

# load data
reference_absorptivities = pd.read_csv('Absorptivities.csv')
spectra = pd.read_csv('Absorption_spectra_set.csv')
concentration = pd.read_csv('Concentration_set.csv')

# inspect data
print(
    'reference_absorptivities', reference_absorptivities.shape, '\n',
    'spectra', spectra.shape, '\n',
    'concentration', concentration.shape, '\n',
)

spectra.set_index('wavelength').plot(legend=False)

# plot absorbance (y) vs concentration (x) at each wavelength
# shouldn't we expect it ta have a ~linear relationship?
plt.figure()
for i in range(1, spectra.shape[0]):
    plt.scatter(
        concentration['concentration_lactate'],
        spectra.T[i].values[1:]
    )


# ------------- #

# define x and y
y = concentration['concentration_lactate']
spectra_ = spectra.drop('wavelength', axis=1).transpose()

# split data
X_train, X_test, y_train, y_test = train_test_split(spectra_, y, test_size=0.2, random_state=42)

# define model
model = Pipeline([
    # ('scale_each_sample', preprocessing.Normalizer()),
    ('scale', preprocessing.StandardScaler()),
    # ('feature_selection', feature_selection.RFE(linear_model.Ridge(alpha=0.01), n_features_to_select=150)),
    ('reduce dims', PCA(20)),
    # ('add_polynomials', preprocessing.PolynomialFeatures(2, include_bias=False)),
    # ('clf', RandomForestRegressor(n_estimators=10, min_samples_leaf=10)),
    ('clf', linear_model.LinearRegression()),
    # ('clf', MLPRegressor(hidden_layer_sizes=[2], verbose=True, max_iter=200, tol=0.001)),
    # ('clf', linear_model.Ridge(alpha=0.0015)),  # can drop PCA
])

# train
model.fit(X_train, y_train)

# eval
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R-squared: {r2}')
print(model.steps[2][1].coef_)

# ---------------- #

# examine pca
pca = PCA(20)
pd.DataFrame(pca.fit_transform(preprocessing.StandardScaler().fit_transform(X_train)), index=X_train.index).iloc[:4, :].plot()
plt.figure()
plt.plot(pca.fit_transform(X_train))

# random forest feature importance
clf = RandomForestRegressor(n_estimators=5, min_samples_leaf=10)
clf.fit(X_train, y_train)

forest_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
plt.figure()
forest_importances.sort_values(ascending=False)[:50].plot.bar()

# feature permutation importance
importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
plt.figure()
pd.Series(importance.importances_mean, index=X_train.columns).sort_values(ascending=False)[:50].plot.bar()

# inspect scaling
X_train.T.plot()
pd.DataFrame(preprocessing.StandardScaler().fit_transform(X_train)).T.plot()