import pandas as pd
import pylab as plt

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load data
spectra = pd.read_csv('Absorption_spectra_set.csv')
concentration = pd.read_csv('Concentration_set.csv')

# define x and y
y = concentration['concentration_lactate']
spectra_ = spectra.drop('wavelength', axis=1).transpose()
spectra_ = spectra_.div(spectra_.max(axis=1), axis=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(spectra_, y, test_size=0.2, random_state=42)

# alpha params
start = 0.000001
end = 10
num_points = 100
exponential_series = np.exp(np.linspace(np.log(start), np.log(end), num_points))

fig, ax = plt.subplots()
plt.subplots_adjust(top=0.8)
spectra.set_index('wavelength').plot(legend=False, ax=ax)
# (spectra_.T.set_index(spectra['wavelength']) * 10).plot(legend=False, ax=ax)
for alpha in exponential_series:

    # define model
    model = Pipeline([
        ('scale', preprocessing.StandardScaler()),
        ('clf', linear_model.Ridge(alpha=alpha)),
    ])

    # train
    model.fit(X_train, y_train)

    # eval
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # plot most the important wavelengths
    min_value = np.min(model.steps[1][1].coef_)
    max_value = np.max(model.steps[1][1].coef_)
    importance = (model.steps[1][1].coef_ - min_value) / (max_value - min_value)
    note = 'The curve at the bottom show feature importance. \n Higher values -> more important. \n\n'
    pd.Series(importance, index=spectra['wavelength']).plot(
        ax=ax,
        title=f"{note} R2: {r2.round(2)}, alpha: {alpha.round(6)}",
    )
    plt.pause(0.0001)