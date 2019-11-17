import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

#importing the dataset
dataset = pd.read_csv('creditcard.csv')
plt.hist(dataset.Time,range(20))
plt.title('Time Distribution')
plt.show()

#feature scaling our data
from sklearn.preprocessing import StandardScaler