import random
import numpy as np
import pickle
import json
import tensorflow as tf
from tf_keras.models import load_model

model= tf.keras.models.load_model(
    "chatbot_model.keras", custom_objects=None, compile=True, safe_mode=True
)
import nltk
nltk.download("WordNet")
nltk.download("punkt")

from nltk.stem import WordNetLemmatizer

import tensorflow as tf



lemmatizer= WordNetLemmatizer()
with open("intents.json", "r") as file:
    intents = json.load(file)

# Load the trained model


# Load preprocessed data
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

print("Done")







def clean_up_sentence(sentence):
    sentence_words= nltk.word_tokenize(sentence) 
    sentence_words= [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag= [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
    return np.array(bag)

print("gO BOT IS RUNNING")

def predict_class(sentence):
    bow= bag_of_words(sentence)
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD= 0.25
    results= [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key= lambda x:x[1], reverse= True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

print("GO")

import random

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json.get('intents', [])  # Ensure list_of_intents is a list or an empty list if 'intents' key is missing
    result = "Sorry, I didn't understand that."  # Default response if no matching tag is found
    for i in list_of_intents:
        if 'tag' in i and i['tag'] == tag:  # Check if 'tag' exists in i
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    ints= predict_class(text)
    res= getResponse(ints, intents)
    return res



#taking input from the user
while True:
    print("Bot: ", end="")
    inp= input()
    if inp.lower()=="quit":
        break
    else:
        print(chatbot_response(inp))
