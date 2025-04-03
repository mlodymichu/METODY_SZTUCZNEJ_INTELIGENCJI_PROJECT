from datetime import datetime
from itertools import count

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

data_set = pd.read_csv("mobile_phone_data_set.csv", delimiter=";")
#print(data_set)
#print(data_set.dtypes)
#print(data_set.describe(include=[np.number]))


data_set.loc[data_set['Device Model'] == 'iPhone 12', "Release date"] = datetime(2020,10,13)
data_set.loc[data_set['Device Model'] == 'Xiaomi Mi 11', "Release date"] = datetime(2022,2,22)
data_set.loc[data_set['Device Model'] == 'Google Pixel 5', "Release date"] = datetime(2020,10,15)
data_set.loc[data_set['Device Model'] == 'Samsung Galaxy S21', "Release date"] = datetime(2021,1,14)
data_set.loc[data_set['Device Model'] == 'OnePlus 9', "Release date"] = datetime(2021,3,21)

data_set.loc[data_set['Device Model'] == 'iPhone 12', "Release price ($)"] = 1199
data_set.loc[data_set['Device Model'] == 'Xiaomi Mi 11', "Release price ($)"] = 963
data_set.loc[data_set['Device Model'] == 'Google Pixel 5', "Release price ($)"] = 699
data_set.loc[data_set['Device Model'] == 'Samsung Galaxy S21', "Release price ($)"] = 799
data_set.loc[data_set['Device Model'] == 'OnePlus 9', "Release price ($)"] = 729

np.savetxt("mobile_phone_data_set_added_data.csv", data_set,delimiter=",", fmt="%s")

#print(data_set.dtypes)
#print(data_set['Release price ($)'].describe(include=[np.number]))
#print(data_set)

#print(data_set.groupby('Device Model')['Gender'].count())

to_plot_gender_device_model = data_set.groupby('Gender')['Device Model'].value_counts()
print(to_plot_gender_device_model)

fig, axs = plt.subplots(1,2)
axs[1]= data_set[data_set["Gender"]=='Female'].groupby('Gender')['Device Model'].value_counts().plot.bar()
axs[0]= data_set[data_set["Gender"]=='Male'].groupby('Gender')['Device Model'].value_counts().plot.bar()
plt.show()

# check_size= data_set[data_set["Gender"]=='Female'].groupby('Gender')['Device Model'].value_counts()
# fig, axs = plt.subplots(1,2)
# axs[0] = plt.bar(data_set[data_set["Gender"]=='Female'].groupby('Gender')['Device Model'].value_counts(),range(check_size.size))
# plt.show()