from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
import xgboost
import nltk
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__,template_folder="template")

@app.route('/')
def hello_world():
    return 'Hey! Please Add index'


@app.route('/index')
def index():
    return flask.render_template('Prediction.html')


open_file = open('modi_word_files.pkl', "rb")
loaded_list = pickle.load(open_file)
open_file.close()

token_corr=loaded_list[2]
token_eng=loaded_list[3]
vocab_size_eng=loaded_list[4]
vocab_size_corr=loaded_list[5]


# Defining Custom Loss Function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

@tf.function
def loss_function(real, pred):
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention
    # optimizer = tf.keras.optimizers.Adam()
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

# Loading trained model
best_model = tf.keras.models.load_model('best_model_general', custom_objects={"loss_function": loss_function})


@app.route('/predict', methods=['POST'])
def predict():

    to_predict_list = request.form.to_dict() #input taken from form
    cus_loc_ven = to_predict_list['text_prediction']

    model=best_model
    UNITS=200
    input_sentence= str(cus_loc_ven)

    # Tokenizing and Padding the sentence
    inputs = [token_corr.word_index.get(i, 0) for i in input_sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = 38, padding = 'post')
    inputs = tf.convert_to_tensor(inputs)
    

    # Initializing result string and hidden states
    result = ''
    hidden = tf.zeros([1, UNITS]), tf.zeros([1, UNITS])

    # Getting Encoder outputs
    enc_out, state_h, state_c = model.encoder([inputs, hidden])
    
    dec_hidden = [state_h, state_c]
    dec_input = tf.expand_dims([token_eng.word_index['<start>']], 0)

    for t in range(38):
        # Getting Decoder outputs fot timestep t
        output, state_h, state_c = model.decoder.timestepdecoder([dec_input, enc_out, state_h, state_c])
        # Getting token index having highest probability
        predicted_id = tf.argmax(output[0]).numpy()
        # Getting output token
        if token_eng.index_word.get(predicted_id, '') == '<end>':
          break
        else:
            result +=token_eng.index_word.get(predicted_id, '')+' '
            dec_input = tf.expand_dims([predicted_id], 0)

    prediction=str(result)
    
    return jsonify({'The predicted Text is:': prediction})   


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.debug = True
    app.run()