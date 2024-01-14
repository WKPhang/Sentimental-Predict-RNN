# -*- coding: utf-8 -*-
"""
Author: Wei Kit Phang
Date: 03 Jan 2024
"""


# Load library
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense,SimpleRNN,Embedding,Flatten
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# Set the working directory to a specific path
os.getcwd() # check current work directory
os.chdir('C:/YOUR/WORK/DIRECTORY')

train_df = pd.read_csv('Data/train.csv',encoding='latin1');
test_df = pd.read_csv('Data/test.csv',encoding='latin1');

train_df = train_df[['text','sentiment']]
test_df = test_df[['text','sentiment']]

train_df['text'].fillna('',inplace=True)
test_df['text'].fillna('',inplace=True)

def func(sentiment):
    if sentiment =='positive':
        return 0;
    elif sentiment =='negative':
        return 1;
    else:
        return 2;

train_df['sentiment'] = train_df['sentiment'].apply(func)
test_df['sentiment'] = test_df['sentiment'].apply(func)

x_train = np.array(train_df['text'].tolist())
y_train = np.array(train_df['sentiment'].tolist())
x_test = np.array(test_df['text'].tolist())
y_test = np.array(test_df['sentiment'].tolist())

x_train
y_train

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

y_train

tokenizer = Tokenizer(num_words=20000)

tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_test)

len(tokenizer.word_index)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, padding='post', maxlen=35)  # Set maxlen to 35
x_test = pad_sequences(x_test, padding='post', maxlen=35)  

x_train[0]

x_train.shape

model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=5, input_length=35))
model.add(SimpleRNN(32,return_sequences=False))
model.add(Dense(3,activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(title='Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Export RNN model
model_filename = "saved_rnn_model.h5"
model.save(model_filename)

# Load the model from the specified path
from tensorflow.keras.models import load_model
loaded_model = load_model("saved_rnn_model.h5")


# Prediction trial 1
text = "Cold food, colder service, I hate this place unless you enjoy disappointment"

new_text_seq = tokenizer.texts_to_sequences([text])
new_text_padded = pad_sequences(new_text_seq, padding='post', maxlen=35)  # Use the max_len determined during training
predictions = model.predict(new_text_padded)
predicted_class_index = predictions.argmax(axis=-1)
if predicted_class_index[0] == 0:
    print("Postive Sentiment");
elif predicted_class_index[0] == 1:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")

# Prediction trial 2
text = "The ramen was good, i love it"

new_text_seq = tokenizer.texts_to_sequences([text])
new_text_padded = pad_sequences(new_text_seq, padding='post', maxlen=35)  # Use the max_len determined during training
predictions = model.predict(new_text_padded)
predicted_class_index = predictions.argmax(axis=-1)
if predicted_class_index[0] == 0:
    print("Postive Sentiment");
elif predicted_class_index[0] == 1:
    print("Negative Sentiment")
else:
    print("Neutral Sentiment")
