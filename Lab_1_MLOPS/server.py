#%%
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
import re

#%%
app = Flask(__name__)

# Load the model
model = load_model('model.h5')

# Define the clean_text function (or import it if defined elsewhere)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

#write a code to import the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    cleaned_text = clean_text(text)
    
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=500)  # Adjust maxlen to match your model's training
    
    prediction = model.predict(padded_sequences)
    response = {'prediction': 'real' if prediction[0][0] > 0.5 else 'fake'}
    print("Prediction: done")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
