# Q1 plotting data using various libraries like matplotlib, seaborn, etc 

import seaborn as sns
import matplotlib.pyplot as plt

# catplot
# By default, the visual representation will be a jittered strip plot
data_1 = sns.load_dataset("titanic")
sns.catplot(data=data_1, x="age", y="class")
plt.show()

# using kind keyword to plot different types of representation
# kind = "box"
sns.catplot(data=data_1, x="age", y="class", kind="box")
plt.show()

# kind = "boxen"
sns.catplot(data=data_1, x="age", y="class", kind="boxen")
plt.show()

# passing additional keywords arguement
# kind = "violin"
sns.catplot(data=data_1,  x="age", y="class", hue="sex", kind="violin", bw_method=.25, cut=0, split=True)
plt.show()

# kind = "bar"
sns.catplot(data=data_1, x="class", y="survived", kind="bar", col="sex", height=4, aspect=6)
plt.show()

# assigning a variable to col or row will automatically create subplots
# taking col
sns.catplot(data=data_1, x="class", y="survived", col="sex", kind="bar", height=4, aspect=.6)
plt.show()

# taking row
sns.catplot(data=data_1, x="class", y="survived", row="sex", kind="bar", height=4, aspect=.6)
plt.show()

# single-subplot figures
sns.catplot(data=data_1, x="age", y="class", kind="violin", color=".9", inner=None)
plt.show()

g = sns.catplot(data=data_1, x="who", y="survived", col="class", kind="bar", height=4, aspect=.6)
g.set_axis_labels("People","Survival Rate")
g.set_xticklabels(["Men","Women","Children"])
g.set_titles("{col_name}{col_var}")
g.set(ylim=(0,1))
g.despine(left=True)
plt.show()

# kind = "point"
sns.catplot(data=data_1, x="age", y="class", kind="point")
plt.show()

# kind = "box" hue = "sex"
sns.catplot(data=data_1, x="age", y="class", kind="box", hue="sex")
plt.show()

# scatterplot
sns.scatterplot(data=data_1, x="class", y="survived", legend="auto")
plt.show()

# histplot
data_2 = sns.load_dataset("penguins")
sns.histplot(data=data_2, x="flipper_length_mm", kde=True)
plt.show()

sns.histplot(data=data_2, y="flipper_length_mm")
plt.show()

sns.histplot(data=data_2, x="flipper_length_mm", binwidth=3)
plt.show()

sns.histplot(data=data_2)
plt.show()

sns.histplot(data=data_2, x="flipper_length_mm", hue="species", element="poly")
plt.show()




# Q2 data cleaning techniques

# Data cleaning techniques
import pandas as pd

print('1. importing the dataset...')
df_local = pd.read_csv('C:/Users/jiger/OneDrive/Desktop/introToDs/all_data.csv')
print('displayig the first 10 rows of dataset')
print(df_local.head(10))

print('2. initial assessment')
print(df_local.info())
print(df_local.describe())

print('3. checking for misiing values')
missing_values = df_local.isnull().sum()
total_missing = missing_values.sum()
if total_missing > 0:
    print('the total missing values are:',missing_values)
    print(df_local.isnull().any(axis=1))
    df_local.dropna(subset='center_state', inplace=True)
    print(df_local.isnull().sum())
    print("We don't see any missing values now.")
else:
    print("We don't see any missing values now.")

print('4. checking for duplicate values..')
duplicate_values = df_local.duplicated().sum()
if duplicate_values != 0:
    print("We see",duplicate_values,"duplicate values")
else:
    print("We don't have any duplicate values!")

print("We are done with our data pre-processing!")
print("Thank you!")




# Q3 encryption

import pandas as pd

data = pd.read_excel("C:/Users/jiger/OneDrive/Desktop/introToDs/data4.xlsx")
print(data)

exposed_data = pd.DataFrame(columns=data.columns)
encrypted_data = pd.DataFrame(columns=data.columns)

for i in range(0,len(data['Password']),1):
    count = 0
    for j in data.loc[i,'Password']:
        if j != '*':
            count = count + 1
    if count == 0:
        print(data.loc[i,'Username'],": Completely encrypted password.")
    elif count < len(data.loc[i,'Password']):
        print(data.loc[i,'Username'],": Partially encrypted password.")
        new_row = {"Name": data.loc[i,"Name"], "Username": data.loc[i,"Username"], "Age": data.loc[i,"Age"], "Password": data.loc[i,"Password"], "Gender": data.loc[i,"Gender"]}
        exposed_data = exposed_data._append(new_row, ignore_index = True)
    elif count == len(data.loc[i,'Password']):
        print(data.loc[i,'Username'],": Exposed password.")
        new_row = {"Name": data.loc[i,"Name"], "Username": data.loc[i,"Username"], "Age": data.loc[i,"Age"], "Password": data.loc[i,"Password"], "Gender": data.loc[i,"Gender"]}
        exposed_data = exposed_data._append(new_row, ignore_index = True)
    else:
        print(data.loc[i,'Username'],": Encountered error while loading data.")

print("Exposed Data:")
print(exposed_data)
print("Do you want to encrypt the exposed data and merge them in the same order of the existing dataset?")
print("1. Yes \n2. No")
input1 = str(input("")).lower().strip()
poss1 = ["1","1.","yes","1. yes","1.yes"]
poss2 = ["2","2.","no","2. no","2. no"]
for i in poss1:
    if input1 == i:
        for i in range(0,len(data['Password']),1):
            password_length = len(data.loc[i, 'Password'])
            encrypted_password = '*' * password_length 
            n_row = {"Name": data.loc[i,"Name"], "Username": data.loc[i,"Username"], "Age": data.loc[i,"Age"], "Password": encrypted_password, "Gender": data.loc[i,"Gender"]}
            encrypted_data = encrypted_data._append(n_row, ignore_index = True)
        print(encrypted_data)

for i in poss2:
    if input1 == i:
        print("Okay fuck Off bitch!")


# Q4 build any regressio model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the data
housing_data = pd.read_excel("C:/Users/jiger/OneDrive/Desktop/Book1.xlsx", sheet_name='Sheet1')
print(housing_data.head())

# Checking for null values and handling them
if housing_data.isnull().values.any():
    null_columns = housing_data.columns[housing_data.isnull().any()]
    for col in null_columns:
        if housing_data[col].dtype == 'float64' or housing_data[col].dtype == 'int64':
            housing_data[col] = housing_data[col].fillna(housing_data[col].median())

print(housing_data)

# Verify if all null values are handled
if not housing_data.isnull().values.any():
    X_train = housing_data.drop('price', axis='columns')
    Y_train = housing_data['price']
    
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    # Input for prediction
    area1 = float(input("Enter the required area:\n(NOTE: The area of the house is in sq.ft.): "))
    bedrooms1 = float(input("Enter the number of bedrooms: "))
    age1 = float(input("Enter the age of property: "))

    X_test = np.array([[area1, bedrooms1, age1]])
    price1 = model.predict(X_test)
    print(f'Value of the property as per your requirements is Rs. {price1[0]:,.2f}')

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Scatter plot for actual data
    plt.scatter(housing_data['area'], housing_data['price'], color='BLUE', label='Data points')

    # Plotting the regression line
    x_range = np.linspace(housing_data['area'].min(), housing_data['area'].max(), 100).reshape(-1, 1)
    y_pred = model.predict(np.hstack((x_range, np.zeros_like(x_range), np.zeros_like(x_range))))
    plt.plot(x_range, y_pred, color='RED', label='Regression line')

    # Scatter plot for prediction
    plt.scatter([area1], [price1], color='GREEN', label='Prediction')

    # Labels and title
    plt.xlabel('Area in sq.ft.')
    plt.ylabel('Price in Rs.')
    plt.title('Home Price Prediction Model')
    plt.legend(loc='upper left', fontsize='small', title='Legend')
    plt.show()




Q5 & Q6 Skewness, mean, median, mode and sd

import numpy as np
import pandas as pd
import scipy.stats as sts

# Mean
# Calculating mean using numpy
data1 = np.array([1,2,3,4,5,2,8,7,2])
mean1 = np.mean(data1)
for i in data1:
    print(i)
print('Mean of the above data:',mean1)

# Calculating mean using pandas
data2 = {
    "Age" : [18,24,12,36,45,87,53],
    "Weight" : [65,73,32,45,68,48,56]
    }
df1 = pd.DataFrame(data2)
print(df1)
mean21 = df1['Age'].mean()
print('Average age:',mean21)
mean22 = df1['Weight'].mean()
print('Average weight',mean22)

# Calculating standard deviation for the given data
# using numpy
std1 = np.std(data1)
print('Standard deviation for data1 is', std1)

# using pandas
std21 = data1.std()
print(std21)

std22 = df1['Age'].std()
print(std22)

# calculating skewness using SciPy
data_01 = [1000000,0.02,3,4,5,100]
skew1 = sts.skew(data_01)
print('Skewnes  of the given data is',skew1)

skew2 = pd.Series([22,14,15,18,19,8,9,34,30,7]).skew()
print('Skewness of the given data is', skew2)

skew3 = pd.Series(data_01).skew()
print('Skewness of the data is', skew2)

# Calculating median using pandas
median1 = pd.Series(data1).median()
print('Median is',median1)

# Calculating mode using pandas
mode1 = pd.Series(data1).mode()
print('Mode is',mode1)



# Q7

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

data = pd.read_csv("C:/Users/jiger/Downloads/headbrain3.csv") 

linear_model = ols('Brain_weight ~ Head_size', data=data).fit() 

print(linear_model.summary()) 

fig = plt.figure(figsize=(14, 8)) 

fig = sm.graphics.plot_regress_exog(linear_model, 'Head_size', fig=fig) 
plt.show()			

