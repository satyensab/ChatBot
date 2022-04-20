#imports libraries
import nltk
#necessary files for chatbot to work
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

#variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

#cleans data 
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # splits a large sample of text into words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents to the multiple collection of text data
        documents.append((w, intent['tag']))

        # adds tags to the list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# groupes together the different inflected forms of a word so they can be analyzed as a single item and then lowers each word and removes any duplicates 
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sorts the classes alphabetically
classes = sorted(list(set(classes)))
# the documents are the combination between patterns and intents
print (len(documents), "documents")
# the classes represent the intents, computer language
print (len(classes), "classes", classes)
# the vocabulary of the computer language
print (len(words), "unique lemmatized words", words)

#converts a Python object into a json string.
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#now that we have got the neccessary computer responses we now create our training data
training = []

#output array
output_empty = [0] * len(classes)

# the training set - the data we train to predict computer response
for doc in documents:
    # initialize the collection of words
    bag = []
    pattern_words = doc[0]
    # a root word to correlate with related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # adds related words into a bag of words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
  
#turns training data into a numpy array
random.shuffle(training)
training = np.array(training)

# the train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create models
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile models
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
