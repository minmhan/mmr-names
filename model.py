import os, sys, codecs
import fileinput
import re
import myparser
import numpy as np
import tensorflow.compat.v1 as tf
from random import sample
#import keras
#from keras.layers import concatenate, Dense, Embedding
import keras
from keras.layers import concatenate, Dense, Embedding
tf.disable_v2_behavior()

MAX_LENGTH = 15
start_and_pad_tokens = ['^', '#']
pad_token = '#'
start_token = "^"
rnn_num_units = 128  # size of hidden state
embedding_size = 64  # for characters
names = []
tokens = []
with open("data/processed-text/2019/YG.txt", "r", encoding="utf-8") as file:
    for line in file.readlines():
        p = myparser.MyParser()
        tokens.append(p.syllable(line.strip()))
        names.append(line.strip())


tokens.append(start_and_pad_tokens)
token_to_id = {k:v for v,k in enumerate(set([item for sublist in tokens for item in sublist]))}
id_to_token = list(set([item for sublist in tokens for item in sublist]))
print(len(token_to_id))

n_tokens = len(token_to_id)


def to_matrix(names_tokens, max_len=None, pad=token_to_id[pad_token], dtype=np.int32):
    """Casts a list of names into rnn-digestable padded matrix"""
    max_len = max_len or max(map(len, names_tokens))
    names_ix = np.zeros([len(names_tokens), max_len], dtype) + pad

    for i in range(len(names_tokens)):
        #print(names_tokens[i])
        name_ix = list(map(token_to_id.get, names_tokens[i]))
        names_ix[i, :len(name_ix)] = name_ix

    return names_ix

embed_x = Embedding(n_tokens, embedding_size)
get_h_next = Dense(rnn_num_units, activation="tanh")
get_probas = Dense(n_tokens, activation="softmax")


def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces
    probabilities for next token x_t+1 and next state h_t+1
    given current input x_t and previous state h_t.
    We'll call this method repeatedly to produce the whole sequence.

    You're supposed to "apply" above layers to produce new tensors.
    Follow inline instructions to complete the function.
    """
    # convert character id into embedding
    x_t_emb = embed_x(tf.reshape(x_t, [-1, 1]))[:, 0]
    # concatenate x_t embedding and previous h_t state
    x_and_h = concatenate([x_t_emb, h_t])
    # compute next state given x_and_h
    h_next = get_h_next(x_and_h)
    # get probabilities for language model P(x_next|h_next)
    output_probas = get_probas(h_next)
    return output_probas, h_next


input_sequence = tf.placeholder(tf.int32, (None, MAX_LENGTH))  # batch of token ids
batch_size = tf.shape(input_sequence)[0]

predicted_probas = []
h_prev = tf.zeros([batch_size, rnn_num_units])  # initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[:, t]  # column t
    probas_next, h_next = rnn_one_step(x_t, h_prev)

    h_prev = h_next
    predicted_probas.append(probas_next)
    # print(probas_next)

# combine predicted_probas into [batch, time, n_tokens] tensor
predicted_probas = tf.transpose(tf.stack(predicted_probas), [1, 0, 2])
print(predicted_probas)
# next to last token prediction is not needed
predicted_probas = predicted_probas[:, :-1, :]
print(predicted_probas)

# flatten predictions to [batch*time, n_tokens]
predictions_matrix = tf.reshape(predicted_probas, [-1, n_tokens])
print(predictions_matrix)
# flatten answers (next tokens) and one-hot encode them
answers_matrix = tf.one_hot(tf.reshape(input_sequence[:, 1:], [-1]), n_tokens)
loss = tf.reduce_mean(keras.losses.categorical_crossentropy(answers_matrix, predictions_matrix))
optimize = tf.train.AdamOptimizer().minimize(loss)


tf.disable_eager_execution()
s = tf.compat.v1.Session()
s.run(tf.global_variables_initializer())

batch_size = 32
history = []

for i in range(1000):
    batch = to_matrix(sample(tokens, batch_size), max_len=MAX_LENGTH)
    loss_i, _ = s.run([loss, optimize], {input_sequence: batch})

    history.append(loss_i)
    print(loss_i)

    # if (i + 1) % 100 == 0:
    #     clear_output(True)
    #     plt.plot(history, label='loss')
    #     plt.legend()
    #     plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge"

for i in range(3):
    batch = to_matrix(sample(tokens, batch_size), max_len=MAX_LENGTH)

x_t = tf.placeholder(tf.int32, (1,))
h_t = tf.Variable(np.zeros([1, rnn_num_units], np.float32))  # we will update hidden state in this variable
next_probs, next_h = rnn_one_step(x_t, h_t)


def generate_sample(seed_phrase=start_token, max_length=MAX_LENGTH):
    '''
    This function generates text given a `seed_phrase` as a seed.
    Remember to include start_token in seed phrase!
    Parameter `max_length` is used to set the number of characters in prediction.
    '''
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t, h_t.initial_value))
    #print("in generate")
    # feed the seed phrase, if any
    for ix in x_sequence[:-1]:
        s.run(tf.assign(h_t, next_h), {x_t: [ix]})

    # start generating
    for _ in range(max_length - len(seed_phrase)):
        x_probs, _ = s.run([next_probs, tf.assign(h_t, next_h)], {x_t: [x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens, p=x_probs[0]))

    return ''.join([id_to_token[ix] for ix in x_sequence if id_to_token[ix] != pad_token])

# without prefix
for _ in range(30):
    print(generate_sample(seed_phrase=["^", "မ"])[1:])