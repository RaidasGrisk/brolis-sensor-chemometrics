'''
YT video to help understand the measurement process: https://youtu.be/CEUa1JgKcp0?si=u4uqtjKKAl3LA62J

Notes:

    ▪ Using ridge regression (regularize by keeping the squares of coefficients as low as possible), we can see which
    wavelengths are most used by the model to predict y. Some ranges are more useful than others. e.g. ~2110 and ~2300.
    Run ridge_feature_importance.py for visualization.

    ▪ To get reasonable results must either use PCA or regularize heavily to reduce X dim. RandomForest,
    while not the best algo for this, could be used to pick features. Does not work as features are too
    similar and signal is too hidden.

    ▪ How do I interpret PCA that is done across the wavelengths? Specific PCA components shows very high (~70 to 95%)
    correlation with target concentration. Is it just a coincidence that 4th component works so well, while the rest
    are not adding much to the signal..? Using PCA's 4th component as a single feature yields 0.91 R2 using LR.

    ▪ Too little data for RNN? No. There's no signal in the sequence of absorptivity across different wavelengths.
    The signal lies in specific wavelengths, not the sequence. Check out the X (after post-process) vs Y plot -
    if some ranges are identical - there's no predicting power in these. The signal lies in ranges, where each
    sample diverge from the rest and not overlap. Transformer should work well..?

    ▪ After data processing, X samples (absorption at different wavelengths across measurements) are nearly identical.
    Must engineer new features that would better differentiate each sample and amplify the differences. This would prob
    yield best model results - how do we engineer new features that amplify differences at specific wavelengths?
    PCA does it for us, but can we come up with even better features?
        - Divide absorptions by max / min - spectra_.div(spectra_.max(axis=1), axis=0) -> [1, 0.98, 0.97 ...]

    ▪ Instead of scaled raw absorbance data at each wavelength, do % change in between subsequent wavelengths.
    This completely does not work and I've no understanding why. Does absorbance intercept manner that much? No.
    But this method does not amplify the slight deviations in absorbance that actually matter for concentration.
    spectra_ = spectra_.pct_change(axis=1).drop(0, axis=1)

    ▪ For now it's not about the model architecture. It's about intuitive understanding of the data and
    why PCA works so well, while other methods yield little results. How do I explain and visualize this?

Model results:                            test set R-squared
    Linear + PCA(20) + manual max scale - 0.943
    Linear + PCA(20)                    - 0.913
    Ridge (alpha=0.0015)                - 0.894
    PCA(5) + Poly(2) + Ridge            - 0.924
    PCA(80) + CNN                       - 0.922
    CNN no PCA                          - ~0.8
    RNN - couldn't get it to work properly

'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.svm import SVR
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
# as in the video: https://youtu.be/CEUa1JgKcp0?si=wwjOgg2gzEaqHZPi&t=272
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
spectra_ = spectra_.div(spectra_.max(axis=1), axis=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(spectra_, y, test_size=0.2, random_state=42)

# define model
model = Pipeline([
    # ('scale_each_sample', preprocessing.Normalizer()),
    # ('scale', preprocessing.StandardScaler()),
    ('reduce dims', PCA(20)),
    # ('add_polynomials', preprocessing.PolynomialFeatures(2, include_bias=False)),
    ('clf', linear_model.LinearRegression()),
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
print(model.steps[1][1].coef_)

# ---------------- #

# examine pca
pca = PCA(20)
pd.DataFrame(pca.fit_transform(X_train), index=X_train.index).iloc[:, :4].plot()
plt.figure()
plt.plot(X_train)

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