from datetime import datetime
from itertools import count

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

data_set = pd.read_csv("mobile_phone_data_set_new.csv", delimiter=";")
#print(data_set)
#print(data_set.dtypes)
#print(data_set.describe(include=[np.number]))

print(data_set.iloc[:,1:].describe(include=[np.number]))

#print(data_set.dtypes)
#print(data_set['Release price ($)'].describe(include=[np.number]))
#print(data_set)

#print(data_set.groupby('Device Model')['Gender'].count())

each_model = set()
for x in data_set['Device Model']:
    each_model.add(x)
print(each_model)
indx = np.arange(len(each_model))
print(indx)

# to_plot_gender_device_model = data_set.groupby('Gender')['Device Model'].value_counts()
# print(to_plot_gender_device_model)


#TO JEST WYKRES MARKI->PLEC
bar_width = 0.35

fig1, ax = plt.subplots(figsize=(10,12))
barFemale = ax.bar(indx - bar_width/2, data_set[data_set["Gender"]=='Female'].groupby('Gender')['Device Model'].value_counts(), bar_width, label='Female')
barMale = ax.bar(indx + bar_width/2, data_set[data_set["Gender"]=='Male'].groupby('Gender')['Device Model'].value_counts(), bar_width, label='Male')
ax.set_xticks(indx)
ax.set_xticklabels(each_model,rotation=45)
ax.legend()
plt.show()

color_dict = {'iPhone 12':'red', 'Xiaomi Mi 11':'blue', 'Google Pixel 5':'black', 'Samsung Galaxy S21':'green','OnePlus 9':'purple'}

fig2, ax = plt.subplots()
ax.scatter(data_set['App Usage Time (min/day)'],data_set['User ID'], c=[color_dict[i] for i in data_set['Device Model']])
plt.show()