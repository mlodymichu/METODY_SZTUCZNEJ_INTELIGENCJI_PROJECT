from datetime import datetime
from itertools import count

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

os.environ["LOKY_MAX_CPU_COUNT"] = "4"   # <- np. 4, jeśli masz 4 rdzenie, albo 8


# Ustawienia wyświetlania
pd.set_option('display.max_columns', None)

data_set = pd.read_csv('mobile_phone_data_set_new.csv', delimiter=';')
for col in ['clock_speed', 'm_dep']:
     data_set[col] = data_set[col].str.replace(',', '.').astype(float)

#Poszczegolne wizualizacje uzyte we wstepie
binary_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
all_features_present = (data_set[binary_features] == 1).all(axis=1)
filtered_df = data_set[all_features_present]
result = filtered_df.groupby('price_range').size()

data_set['clock_speed'] = pd.to_numeric(data_set['clock_speed'], errors='coerce')
data_to_plot = [data_set[data_set['price_range'] == i]['clock_speed'].dropna()
                for i in sorted(data_set['price_range'].unique())]

plt.figure(figsize=(8, 5))
plt.boxplot(data_to_plot, tick_labels=sorted(data_set['price_range'].unique()), vert=True)
plt.title("Zależność clock_speed od price_range")
plt.xlabel("Przedział cenowy")
plt.ylabel("Częstotliwość procesora (GHz)")
plt.grid(True)
plt.show()

median_battery = data_set.groupby('price_range')['battery_power'].median()
plt.figure(figsize=(8, 5))
plt.plot(median_battery.index, median_battery.values, marker='o', linestyle='-')
plt.xlabel("Przedział cenowy")
plt.ylabel("Mediana battery_power (mAh)")
plt.title("Mediana battery_power w zależności od price_range")
plt.grid(True)
plt.show()

data_set['m_dep'] = pd.to_numeric(data_set['m_dep'], errors='coerce')
srednia = data_set['m_dep'].mean()

mean_cores = data_set.groupby('price_range')['n_cores'].mean()
plt.figure(figsize=(8, 5))
plt.plot(mean_cores.index, mean_cores.values, marker='o')
plt.title("Średnia liczba rdzeni procesora (n_cores) w zależności od price_range")
plt.xlabel("Przedział cenowy")
plt.ylabel("Średnia liczba rdzeni")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
scatter = plt.scatter(data_set['px_height'], data_set['px_width'],c=data_set['price_range'])
plt.title("Przedział cenowy w zależności od wielkości telefonu")
plt.xlabel("Wyokość telefonu w piksleach")
plt.ylabel("Szerokość telefonu w piksleach")
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Przedział cenowy")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='price_range', y='pc', data=data_set, vert=True)
plt.title('Zależność jakości aparatu tylnego (pc) od price_range')
plt.xlabel('Przedział cenowy')
plt.ylabel('Jakość aparatu tylnego (pc)')
plt.show()

mean_ram_per_price = data_set.groupby('price_range')['ram'].mean().reset_index()
mean_ram_per_price = mean_ram_per_price.sort_values('price_range')
plt.figure(figsize=(8,6))
plt.plot(mean_ram_per_price['price_range'], mean_ram_per_price['ram'], marker='o', linestyle='-')
plt.title('Średnia wartość RAM dla poszczególnych przedziałów cenowych')
plt.xlabel('Przedział cenowy (price_range)')
plt.ylabel('Średnia RAM')
plt.grid(True)
plt.xticks(mean_ram_per_price['price_range'])
plt.show()

#Tworzenie nowego data setu z jedna nowa data
data_set_added_one_attribute = data_set
data_set_added_one_attribute['Ram_to_battery_ratio']=data_set_added_one_attribute['ram']/data_set_added_one_attribute['battery_power']
data_set_added_one_attribute.insert(20, 'Ram_to_battery_ratio', data_set_added_one_attribute.pop('Ram_to_battery_ratio'))
np.savetxt('mobile_phone_data_set_new_added_one_attribute.csv',data_set_added_one_attribute,delimiter=',',fmt='%s')

#Tworzenie nowego data setu z dwoma nowymi data
data_set_added_two_attributes = data_set_added_one_attribute
data_set_added_two_attributes['resolution_product'] = data_set_added_two_attributes['px_height'] * data_set_added_two_attributes['px_width']
data_set_added_two_attributes.insert(21, 'resolution_product', data_set_added_one_attribute.pop('resolution_product'))
np.savetxt('mobile_phone_data_set_new_added_two_attributes.csv',data_set_added_two_attributes,delimiter=',',fmt='%s')


data_set = data_set.to_numpy()
data_set_added_one_attribute = data_set_added_one_attribute.to_numpy()
data_set_added_two_attributes = data_set_added_two_attributes.to_numpy()

X_base = data_set[:, :-1]
y_base = data_set[:, -1]

X_one_added = data_set_added_one_attribute[:, :-1]
y_one_added = data_set_added_one_attribute[:, -1]

X_two_added = data_set_added_two_attributes[:, :-1]
y_two_added = data_set_added_two_attributes[:, -1]

classifiers = {
    "Gaussian": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "MPLClassifier": MLPClassifier(),
    "SVMClassifier": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
}

rkf = RepeatedKFold(n_splits=2, n_repeats=5)
wynikiBase = np.zeros((5,10))
for i, (train_index, test_index) in enumerate(rkf.split(X_base, y_base)):
    X_train_base, X_test_base = X_base[train_index], X_base[test_index]
    y_train_base, y_test_base = y_base[train_index], y_base[test_index]
    for clf_id, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X_train_base, y_train_base)
        y_pred_base = clf.predict(X_test_base)
        accuracy_base = accuracy_score(y_pred_base, y_test_base)
        wynikiBase[clf_id, i]= accuracy_base

sredniaBase = np.mean(wynikiBase,axis=1)
stdBase = np.std(wynikiBase,axis=1)
print("--------------------------------------------------")
print("Wyniki dla początkowego zbioru danych")
for name, mean, std in zip(classifiers.keys(), sredniaBase, stdBase):
    print(f"{name}: Średnia = {mean:.3f}, Odchylenie std = {std:.3f}")


wynikiOneAdded = np.zeros((5,10))
for i, (train_index, test_index) in enumerate(rkf.split(X_one_added, y_one_added)):
    X_train_one_added, X_test_one_added = X_one_added[train_index], X_one_added[test_index]
    y_train_one_added, y_test_one_added = y_one_added[train_index], y_one_added[test_index]
    for clf_id, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X_train_one_added, y_train_one_added)
        y_pred_one_added = clf.predict(X_test_one_added)
        accuracy_one_added = accuracy_score(y_pred_one_added, y_test_one_added)
        wynikiOneAdded[clf_id, i]= accuracy_one_added

sredniaOneAdded = np.mean(wynikiOneAdded,axis=1)
stdOneAdded = np.std(wynikiOneAdded,axis=1)
print("--------------------------------------------------")
print("Wyniki dla zbioru danych z jedną nową kolumnamą")
for name, mean, std in zip(classifiers.keys(), sredniaOneAdded, stdOneAdded):
    print(f"{name}: Średnia = {mean:.3f}, Odchylenie std = {std:.3f}")

wynikiTwoAdded = np.zeros((5,10))
for i, (train_index, test_index) in enumerate(rkf.split(X_two_added, y_two_added)):
    X_train_two_added, X_test_two_added = X_two_added[train_index], X_two_added[test_index]
    y_train_two_added, y_test_two_added = y_two_added[train_index], y_two_added[test_index]
    for clf_id, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X_train_two_added, y_train_two_added)
        y_pred_two_added = clf.predict(X_test_two_added)
        accuracy_two_added = accuracy_score(y_pred_two_added, y_test_two_added)
        wynikiTwoAdded[clf_id, i]= accuracy_two_added

sredniaTwoAdded = np.mean(wynikiTwoAdded,axis=1)
stdTwoAdded = np.std(wynikiTwoAdded,axis=1)
print("--------------------------------------------------")
print("Wyniki dla zbioru danych z dwoma nowymi kolumnami")
for name, mean, std in zip(classifiers.keys(), sredniaTwoAdded, stdTwoAdded):
    print(f"{name}: Średnia = {mean:.3f}, Odchylenie std = {std:.3f}")