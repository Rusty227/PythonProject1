#making a magic 8 ball , may need a defined set, a whille true for continues play, exit command, random evaluator
#issue here was i wanted to make a random generator, but i did  not import the premade program
import random

import time
import sys

def question():
    return ["tough luck", "tryagain", "yes", "no"]  # issue here was i was making a list, but without return it was returning nothing

def slow_type(text, delay = 0.5):  # create function that will contain text and delay time
    for char in text:
        print(char, end='', flush=True)  # for the text in the function end with a space and then clear off the screen
        time.sleep(delay) # on the imported function add the sleep time via delay recall
    print()  # print the end of the message



print("type in your question")


while True:
    x = input()

    if not x.strip():  #issue here i was trying to make it if something was not written then write this instead. i left IF as a TRUE and IF NOT is a FALSE
        #this is also checking if nothing is typed
        print("type something")


    elif x.lower() == "quit":
        break



    else:
        slow_type("thinking.....", delay=0.2)  # decided to have fun and slow it down for suspence
        time.sleep(2)
       # print(random.choice(question())) old instant print
        #in here random is the generator, and choice is give the program the choice/free will to pick from the list
        slow_type(random.choice(question()), delay=0.5)   #given as im returning 1  thing to the function i need to add the delay so 2 things get returned

    again = input("would you like to try again YES / No")

    if again.lower() not in ["no", "quit"]:  # i wanted to make it a list of words not a string so needed to use []
        continue
    break