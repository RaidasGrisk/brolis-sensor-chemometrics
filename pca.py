import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

# load data
spectra = pd.read_csv('Absorption_spectra_set.csv')
concentration = pd.read_csv('Concentration_set.csv')

# define x and y
y = concentration['concentration_lactate']
spectra_ = spectra.drop('wavelength', axis=1).transpose()

# do PCA
scaler = preprocessing.StandardScaler()
pca = PCA(20)

# preprocess
data = scaler.fit_transform(spectra_)
data = pd.DataFrame(pca.fit_transform(data))

# check this out (3rd component)
data.corrwith(y, axis=0)