import random

import numpy as np
import time
import pandas as pd







temps = np.linspace(15, 25, 50)  # 50 values between 15 and 25
# Your code goes here
#print ("temp",temps)


noise = np.random.random(50) * 1 - 0.5
#print ("noise",noise)

noisytemp = temps + noise
print ("noisytemp",noisytemp[-2:])  #give me the last 2 numbers ignore the rest
# and :-2 is ginore the lst 2 numbers so in term
# where : is the ones before or after is ignored

sample= np.random.choice(noisytemp, 10, replace=False)
print("sample",sample)  #  PC chooses random 10 samples of noicytemps
# the reason why it is false is that it does not  picjk the same number twice

nosy =  np.random.choice(noise, 10, replace=False)
print("nosy",nosy) 







OneDimensionalArray = np.array([[1, 2, 3, 4, 5, 6]])
TwoDimensionalArray = np.array([[1, 2, 3], [4, 5, 6]])
#print("onedimensional", OneDimensionalArray)
#print("two dimensional", TwoDimensionalArray)

sequence = np.arange(10) # doing the whoel intervals
#print("sequence", sequence)
sequence2 = np.arange(1,20,2)  # start at 1, up to 20, in the interval of 2
#print("sequence", sequence2)



feature = np.arange(6, 21)# write your code here
#print("feature",feature)

label = np.array(3 * feature) + 4   # write your code here
#print("label",label)

feature = np.arange(6, 21)
#print(feature)
label = (feature * 3) + 4
#print(label)


noise = (np.random.random([15]) * 4) - 2
#print(noise)
label = label + noise
#print(label)

noise = np.random.random(([2]) *4) -2   # write your code here
#print("noise", noise)
label = label + noise# write your code here
#print(label)
