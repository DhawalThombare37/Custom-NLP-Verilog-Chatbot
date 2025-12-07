import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import os
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize NLTK's WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
data_file = open('D:\DPT Codes\Chatbot-app2\data2.json', encoding='utf-8').read()
intents = json.loads(data_file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process each intent and its patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))
        # Add intent tag to classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files for later use
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Create training data
training = []
output_empty = [0] * len(classes)

# Bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Convert bag and output_row to NumPy arrays
    bag_array = np.array(bag)
    output_row_array = np.array(output_row)

    # Append to training list as tuple
    training.append((bag_array, output_row_array))

# Shuffle training data
random.shuffle(training)

# Convert training list to NumPy array
training_data = np.array(training, dtype=object)

# Separate features and labels
train_x = np.array([i[0] for i in training_data])
train_y = np.array([i[1] for i in training_data])

print("Training data created")

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model using SGD optimizer with learning rate
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit and save the model
epochs = 200
batch_size = 5
verbose = 1

history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

# Ensure the directory exists for saving the model
save_dir = 'D:\DPT Codes\Chatbot-app2'  # Adjust as per your directory structure
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the model to the specified path
model.save(os.path.join(save_dir, 'model.h5'))

print("Model created and saved at:", os.path.join(save_dir, 'model.h5'))

