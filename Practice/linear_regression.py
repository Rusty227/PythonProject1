#@title Code - Load dependencies

#general
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

# @title
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
#so you can have panda just read the data set or form from online just as easy


#@title Code - Read dataset

# Updates dataframe to use specific columns.
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
#grabbing the specifict colums of that data set

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
print(training_df.head(200))   # code on site was wrong. even though it was correct. without the print command it will not show

#print(training_df.iloc[[2]])  having fun printing just the second row
#print(training_df.head(3)) having fun printingthe first 3 rows


# What is the maximum fare?  .max()
# What is the mean distance across all trips? .mean()
# How many cab companies are in the dataset? .nunique()
# What is the most frequent payment type? .value_counts()         (You can combine it with .idxmax() to find the most frequent one.)
# Are any features missing data? .isnull()    .sum()


# Fare_max = max(training_df['FARE'])   this uses the pythin build in and is not the proper panda way.. still works though
# print(" max fare", Fare_max)

print('Total number of rows: {0}\n\n'.format(len(training_df.index)))  # {0} is a place holder forcalculation. format the length/howmany rows are in total in doc and place back in {0}
training_df.describe(include='all') # summarize the chart and print it out. describe only works at the end of the code. otherwise it needs to be wrapped in a print ()


Fare_max = training_df['FARE'].max()   # just gettomg tje ,max number of the fare
print("max fare:", Fare_max)


Mean_max = training_df['TRIP_MILES'].mean()  # getting the mean  of the trip
print("Mean fare:", Mean_max)

Uniquie_Cab =  training_df['COMPANY'].nunique()  #unique() returns a list of all individual company, Nunique returns the numeric amount of companies
print("Uniquie_Cab:", Uniquie_Cab)

Common_Payment = training_df['PAYMENT_TYPE'].value_counts().index[0]  # assing common payment the what it is text wise
Common_Payment_Amount = training_df['PAYMENT_TYPE'].value_counts().iloc[0] # this is to view the most mmon payment number  THIS IS A COUNT
print("Common Payment:", Common_Payment, "||Amount:", Common_Payment_Amount)

# alternatively
max = Common_Payment = training_df['PAYMENT_TYPE'].value_counts().idxmax()  # for readability of name that im only getting the max amount name
print(max)


# missing_Data = training_df.isnull().sum()
# # # print("missing Data: \n" , missing_Data)
# book has it like this
missing_Data = training_df.isnull().sum().sum()  # gather the missing that that shows as blank total, and give me only the total number
print("Missing data:", "No" if missing_Data == 0 else "Yes")  # give me the missing that, if there is make a yezss


max = Common_Payment = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print(max)


#@title Code - View correlation matrix
training_df.corr(numeric_only = True)
print(training_df.corr(numeric_only = True))  # so in here we are seeing who cas the most and least corrdilation, only view one side for ease of use

print(training_df['TIP_RATE'].describe())  # having fun with describe. gives all the numeric values