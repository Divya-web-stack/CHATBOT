import random 
import numpy as np

import tensorflow as tf
import pickle
import json
from tensorflow import keras
from keras import layers
class SGD:
    def __init__(self, lr=0.001, momentum= 0.9, tol=1e-3, nesterov= True):
        self.learning_rate = lr
        
        self.momentum = momentum
        self.tolerance = tol
        self.weights = None
        self.bias = None

class SGD:
    def __init__(self, lr=0.001, momentum=0.9, tol=1e-3, nesterov=True):
        self.learning_rate = lr
        self.momentum = momentum
        self.tolerance = tol
        self.weights = None
        self.bias = None



import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
intents= json.loads(open("intents.json").read())

words=[]
classes=[]
documents=[]
ignored=['?',',','.','!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w= nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words= [lemmatizer.lemmatize(word) for word in words if word not in ignored]
words= sorted(set(words))
classes= sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern_words= doc[0]
    pattern_words= [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row= list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])



random.shuffle(training)

training= np.array(training, dtype = object)
train_x= list(training[:,0])
train_y= list(training[:,1])

x= np.array(train_x)
y= np.array(train_y)



#adding layers  
model = tf.keras.models.Sequential()
print("Done")

# Corrected input shape
#model.add(layers.Dense(128, input_shape=(len(train_x[0])), activation="relu"))

model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))  # Corrected layer syntax
model.add(layers.Dropout(0.5))  # Added layers.
model.add(layers.Dense(len(train_y[0]), activation="softmax"))

# Corrected optimizer
sgd = SGD(lr=0.01, tol=1e-6, momentum=0.9, nesterov=True)

print()
print("Done")

# Corrected loss function and optimizer
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))

print("Done")

model.fit(x, y, epochs=200, batch_size=5)

tf.keras.models.save_model(
    model, "chatbot_model.keras", overwrite=True, 
)

#tf.keras.models.save_model("chatbot_model.h5")  # Changed save format to h5


print("Done")





