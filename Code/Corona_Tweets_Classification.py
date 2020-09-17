#Packages installation
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Data reading
train = pd.read_csv('/Users/spavot/Documents/Perso/Text Classification/Data/Corona_NLP_train.csv', encoding = 'latin')
test = pd.read_csv('/Users/spavot/Documents/Perso/Text Classification/Data/Corona_NLP_test.csv', encoding ='latin')

#Data exploration:

train.info()
#We see that we have 41157 values but we have only 32k non null values for location, we will have to fix this:

#UserName and ScreenName are id related data, we won't use it

#Let's see which location is the more popular
location = train.Location
location = pd.DataFrame(location)
location['Count'] = 1
location = location.groupby('Location').sum().sort_values(by = 'Count', ascending = False).nlargest(15,['Count'])
location = location.reset_index()
sns.barplot(x = 'Count', y = 'Location', data = location)
plt.show()
#We can see that we have some noises and some location are country where other are cities
