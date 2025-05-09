import random

import numpy as np
import time
import pandas as pd





my_colum_names = ['Elenor','chidi','ttahani', 'Jason']

my_data = np.random.randint(0, 101, size=(3 , 4))

#print(my_data)

#print(my_names)

my_dataframe = pd.DataFrame(my_data, columns=my_colum_names)
print(my_dataframe, '\n')

print(my_dataframe[['Elenor']], '\n')

#missed this out as i misstraslated what the question asked
#this is selecting elenaor colum row  1
print("\nSecond row of the Elenor column: %d\n" % my_dataframe['Elenor'][1])


my_dataframe ['Janet'] =  (my_dataframe['ttahani'] + my_dataframe['Jason'])




print(my_dataframe, '\n')

