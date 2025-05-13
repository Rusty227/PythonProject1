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

import matplotlib.pyplot as plt  # to show the graph made by seaborn



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
print(training_df.corr(numeric_only = True))  # so in here we are seeing who cas the most and least correlation, only view one side for ease of use

print(training_df['TIP_RATE'].describe())  # having fun with describe. gives all the numeric values



# showing chart in graph form.
sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars= ["FARE", "TRIP_MILES", "TRIP_SECONDS"])
print("plotting pair plot \n")


#making a clean version with all NA  NULL removed

clean_df = training_df[["FARE", "TRIP_MILES", "TRIP_SECONDS"]].dropna()
sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars= ["FARE", "TRIP_MILES", "TRIP_SECONDS"])



#@title Define plotting functions, not nececssary, fun to look into
#this is for the chart that will show visually the model
def make_plots(df, feature_names, label_name, model_output, sample_size=200):

  random_sample = df.sample(n=sample_size).copy()
  random_sample.reset_index()
  weights, bias, epochs, rmse = model_output

  is_2d_plot = len(feature_names) == 1
  model_plot_type = "scatter" if is_2d_plot else "surface"
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("Loss Curve", "Model Plot"),
                      specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

  plot_data(random_sample, feature_names, label_name, fig)
  plot_model(random_sample, feature_names, weights, bias, fig)
  plot_loss_curve(epochs, rmse, fig)

  fig.show()
  return
#this is also for the model that will show the loss curve
def plot_loss_curve(epochs, rmse, fig):
  curve = px.line(x=epochs, y=rmse)
  curve.update_traces(line_color='#ff0000', line_width=3)

  fig.append_trace(curve.data[0], row=1, col=1)
  fig.update_xaxes(title_text="Epoch", row=1, col=1)
  fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

  return

def plot_data(df, features, label, fig):
  if len(features) == 1:
    scatter = px.scatter(df, x=features[0], y=label)
  else:
    scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

  fig.append_trace(scatter.data[0], row=1, col=2)
  if len(features) == 1:
    fig.update_xaxes(title_text=features[0], row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
  else:
    fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

  return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")



#model needed to build and train the machine

#@title Code - Define ML functions

def build_model(my_learning_rate, num_features):
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = keras.Input(shape=(num_features,))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, features, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse


def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

  print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

  num_features = len(feature_names)

  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, features, label, epochs, batch_size)

  print('\nSUCCESS: training experiment complete\n')
  print('{}'.format(model_info(feature_names, label_name, model_output)))
  make_plots(df, feature_names, label_name, model_output)

  return model

print("SUCCESS: defining linear regression functions complete.")

##########################

#learning_rate = 0.01  # how fast to learn, what i found to be the sweet medium of fast stabalization, but slightly higher bias
learning_rate = 0.001
#learning_rate = 0.0001  # how fast to learn , very slow caused a lot of error
# learning_rate = 1.0  # how fast to learn  # 0.0 a .0 is needed to function # loss curve is spagetty, unlreliable too much noise
epochs = 20  #how many times to run
batch_size = 50   #how much info to use
#batch_size = 500   #how much info to use, batch size was too big for its size. needs to be faster at 0.01 it stabalisez
feature_names = ['TRIP_MILES']    #this is the X axis of the chart
label_name = "FARE"   #this is the Y axis of the chart

# learning_rate = 0.001       # ✅ How fast the model updates when learning (smaller = slower but safer learning)
# epochs = 20                 # ✅ How many full passes over the dataset (more epochs = more chances to learn)
# batch_size = 50             # ✅ How many data points to look at before updating the model during each pass
# feature_names = ['TRIP_MILES']   # ✅ The input(s) to the model — what it's using to make predictions (X axis in simple linear graphs)
# label_name = "FARE"         # ✅ The target value the model is trying to predict (Y axis — the "answer")


#assign to model1 = run_experiment script with the following( the data set, the X axis, the Y axis, how fast, how many times, and how much data
model1 = run_experiment(training_df, feature_names, label_name, learning_rate, epochs, batch_size)


# The following variables are the hyperparameters.
# TODO - Adjust these hyperparameters to see how they impact a training run.
learning_rate = 0.001
epochs = 20
batch_size = 50

# Specify the feature and the label.

training_df.loc[:, 'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60  # mess up here is i used iloc(integer location) instead of loc(label location
features = ['TRIP_MILES', 'TRIP_MINUTES']
label = 'FARE'

model_2 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)


#coding of making prediction

#@title Code - Define functions to make predictions

#just making currency a way to show with cash sign all and add a . 2 spaces to the  front
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

#@title Code - Make predictions
#add the valies of tje predict fare traning def with this data
output = predict_fare(model_2, training_df, features, label)
show_predictions(output)  #now plug into this function the output that was define earlier
#based on this model most prediction are not far off. being less than 1$ for most and 5$ off for one
