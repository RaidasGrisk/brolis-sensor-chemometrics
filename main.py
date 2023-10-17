'''
https://www.youtube.com/watch?v=CEUa1JgKcp0&ab_channel=VernierScienceEducation

TODO:
    experiment 1: instead of scaled raw absorbance data at each wavelength, do change in between subsequent wavelength.
    This should do similar job to PCA (?) but better 🤔.
'''

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn import feature_selection

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
plt.scatter(
    concentration['concentration_lactate'],
    spectra.drop('wavelength', axis=1).T[0]
)

# ------------- #


# define x and y
y = concentration['concentration_lactate']
spectra_ = spectra.drop('wavelength', axis=1).transpose()

# split data
X_train, X_test, y_train, y_test = train_test_split(spectra_, y, test_size=0.2, random_state=42)

# define model
model = Pipeline([
    ('scale', preprocessing.StandardScaler()),
    # ('feature_selection', feature_selection.RFE(linear_model.Ridge(alpha=0.5), n_features_to_select=20)),
    ('reduce dims', PCA(20)),
    ('clf', linear_model.LinearRegression()),
    # ('clf', linear_model.Ridge(alpha=0.5)),
    # ('clf', linear_model.Lasso(alpha=10)),
])

# train
model.fit(X_train, y_train)

# eval
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# print(model.steps[1][1].coef_)
