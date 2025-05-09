import random

import numpy as np
import time
import pandas as pd

#creating and populating a 5 x 2 array
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

#create a list that holds two colum names
my_colum_names = ['temperature', 'activity']

# a data frame
my_dataframe = pd.DataFrame(my_data, columns=my_colum_names)



# adding a new colum to the data frame and add 2 more number to it
# this works by adding adusted to after actifity
my_dataframe['adjusted'] = my_dataframe['activity' ] + 2

# this is just a funcies for me. getting the difference that i know is a negative and making it a positive
# my_dataframe['difference'] = ((my_dataframe['activity' ] - my_dataframe['temperature'])* -1)





#print(my_dataframe)

print("row 0 1 2")
print(my_dataframe.head(3))

print ("row 2")
print(my_dataframe.iloc[[2]])

# diufference between [[2]] and [2]
##[2] is wshowing the whole data series of the subset
##[[2]] is showing the colum of the data series

##Use iloc[2] when you want a single row as a Series (for quick value access).

##Use iloc[[2]] when you want to keep the DataFrame structure (e.g. for further filtering or display).

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

print("Column 'temperature':")
print(my_dataframe['temperature'])

