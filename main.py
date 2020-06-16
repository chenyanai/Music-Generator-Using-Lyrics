import Preprocessing
import Model
import pickle
import os
from tensorflow.keras.utils import plot_model


SEQ_LEN = 20
TRAIN_PATH = 'Data/train_data.pickle'
TEST_PATH = 'Data/test_data.pickle'

start_fresh = True
train_model = True
melody_type = 2
generate_required_songs = False

X_train, y_train, x_test, tokenizer, embedding_matrix, vocab_size = \
    Preprocessing.pp_pipeline(data_folder=r'Data', melody_type=melody_type, save_data=start_fresh, seq_len=SEQ_LEN)

print(f'vocab_size: {vocab_size}')

mid_vector_size = len(x_test[0]['melody_vectors'][0][0:5][0])

if train_model:
    if melody_type == 1:
        model = Model.first_melody_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
                                         embedding_matrix=embedding_matrix, vocab_size=vocab_size)
    else:
        model = Model.second_melody_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
                                          embedding_matrix=embedding_matrix, vocab_size=vocab_size)
    # plot_model(model, to_file='new_model_plot.png', show_shapes=True, show_layer_names=True)
    history = Model.train_model(model, X_train, y_train, vocab_size, melody_input_type=melody_type)
else:
    if melody_type == 1:
        model = Model.load("model_eos_trainable_melody1.h5")
    else:
        model = Model.load("model_eos_trainable_melody2.h5")


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
word_map = tokenizer.word_index

for Xi_test in x_test:
    song = Model.generate_song(model, Xi_test, word_dict=word_map, reverse_word_dict=reverse_word_map, melody_type=melody_type)
    print(song)

if generate_required_songs:
    data_folder = r'Data'
    with open(os.path.join(data_folder, f'test_data1.pickle'), 'rb') as f:
        x_test_1 = pickle.load(f)

    with open(os.path.join(data_folder, f'test_data2.pickle'), 'rb') as f:
        x_test_2 = pickle.load(f)

    model_1 = Model.load("model_eos_trainable_melody1.h5")
    model_2 = Model.load("model_eos_trainable_melody2.h5")

    Model.generate_songs_using_first_word(model_1, model_2, x_test_1, x_test_2,
                                          word_dict=word_map, reverse_word_dict=reverse_word_map)