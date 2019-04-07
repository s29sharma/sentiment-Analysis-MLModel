from tensorflow.python.keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing import sequence
word2index = imdb.get_word_index()
from nltk.tokenize import word_tokenize
model = load_model('temp_model.sav')
model._make_predict_function()
from flask import Flask,request,redirect
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    print(request.args.get('text'))
    text = request.args.get('text')
    test = []
    for word in word_tokenize(text):
        if word in word2index:
            test.append(word2index[word])
    x_new = sequence.pad_sequences([test], maxlen=80)
    x = model.predict(x_new)
    val = x[0][0]
    print(x[0][0])
    if((val * 100) < 50):
        return jsonify(val=str(val*100),
                       sentiment="negative")
    else:
        return jsonify(val=str(val * 100),
                       sentiment="positive")


if __name__ == '__main__':
    app.run(debug=False)
