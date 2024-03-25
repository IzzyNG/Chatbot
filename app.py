from flask import Flask, render_template, request, jsonify
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json
import random

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data and model
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hiddent_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']

    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Choose a random response from intents.json
    for intent in intents['intents']:
        if tag == intent["tag"]:
            response = random.choice(intent['responses'])
            return jsonify({'message': response})

    return jsonify({'message': 'I do not understand...'})

if __name__ == '__main__':
    app.run(debug=True)


