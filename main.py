import Preprocessing
import Model
import pickle
from tensorflow.keras.utils import plot_model


SEQ_LEN = 20
TRAIN_PATH = 'Data/train_data.pickle'
TEST_PATH = 'Data/test_data.pickle'

start_fresh = True
train_model = True
melody_type = 2

X_train, y_train, x_test, tokenizer, embedding_matrix, vocab_size = \
    Preprocessing.pp_pipeline(data_folder=r'Data', melody_type=melody_type, save_data=start_fresh, seq_len=SEQ_LEN)

print(f'vocab_size: {vocab_size}')

mid_vector_size = len(x_test[0]['melody_vectors'][0][0:5][0])

if train_model:
    if melody_type == 1:
        model = Model.build_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
                              embedding_matrix=embedding_matrix,vocab_size=vocab_size)
    else:
        model = Model.new_model(sequence_length=SEQ_LEN, mid_data_len=mid_vector_size,
                                embedding_matrix=embedding_matrix, vocab_size=vocab_size)
    # plot_model(model, to_file='new_model_plot.png', show_shapes=True, show_layer_names=True)
    history = Model.train_model(model, X_train, y_train, vocab_size, melody_input_type=melody_type)
else:
    if melody_type == 1:
        model = Model.load("model_eos_trainable_melody1.h5")
    else:
        model = Model.load("model_eos_trainable_melody2_1024b_5e.h5")


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
word_map = tokenizer.word_index

for Xi_test in x_test:
    song = Model.generate_song(model, Xi_test, words_dict=word_map, reverse_word_dict=reverse_word_map, melody_type=melody_type)
    print(song)