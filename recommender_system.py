import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

#importing dataset
dataset = pd.read_csv('train.csv')
songs = pd.read_csv('songs.csv')
test = pd.read_csv('test.csv')
members = pd.read_csv('members.csv')

dataset = dataset.sample(frac=0.001,random_state=0)
dataset = pd.merge(dataset,songs,on='song_id',how='left')
dataset = pd.merge(dataset,members,on='msno',how='left')
dataset = dataset[:2000]
dataset.info()