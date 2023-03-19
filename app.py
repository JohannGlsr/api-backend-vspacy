from flask import Flask, request, jsonify
import string
from keras.models import load_model
import pickle
import spacy

# Initialisation des stopwords
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

def Preprocess_Sentence(Sentence):
    # Enlever la ponctuation
    Sentence = "".join([i.lower() for i in Sentence if i not in string.punctuation])
    # Enlever les chiffres
    Sentence = ''.join(i for i in Sentence if not i.isdigit())
    # Tokenization : Transformer les phrases en liste de tokens (en liste de mots)
    Sentence = nlp(Sentence)
    # Enlever les stopwords
    Sentence = [i.text for i in Sentence if i.text not in stopwords]
    # Enlever les mots qui ne sont pas alphabétiques ou qui ne sont pas dans le dictionnaire
    Sentence = ' '.join(w for w in Sentence if w.isalpha() and w.lower() in nlp.vocab)
    # Lemmatisation
    Sentence = ' '.join([token.lemma_ for token in nlp(Sentence)])
    
    return Sentence 

# Chargement du tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Chargement du modèle
loaded_model = load_model('advance_model.h5')

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    phrase = request.form['phrase']
    sequence = Preprocess_Sentence(phrase)
    sequence = tokenizer.texts_to_sequences([sequence])
    while len(sequence[0]) < 35:
        sequence[0].insert(0, 0)
    prediction = loaded_model.predict(sequence)
    stringprediction = str(prediction[0][0])
    return jsonify({'prediction': stringprediction})

if __name__ == '__main__':
    app.run()
