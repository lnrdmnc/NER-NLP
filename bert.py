import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import bert
import os
import random

# URL del modello Universal Sentence Encoder Multilingual
tf_sentencepiecemodel_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/1"

# Definizione delle variabili mancanti
dataDir = "path_to_data_directory"
max_seq_length = 128
classes = 10
model_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
modelBertDir = "path_to_bert_model_directory"
modelDir = "path_to_save_model_directory"

# Caricamento dei dati
def loadData(tokenizer):
    fileName = os.path.join(dataDir, "data.csv")
    fileTestName = os.path.join(dataDir, "data_test.csv")

    data = []
    data_test = []
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []

    with open(fileName, encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=";")
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                data.append(row)
            line_count += 1

    with open(fileTestName, encoding='utf-8') as csvFileTest:
        csv_reader_test = csv.reader(csvFileTest, delimiter=";")
        line_count = 0
        for row in csv_reader_test:
            if line_count > 0:
                data_test.append(row)
            line_count += 1

    shuffled_set = random.sample(data, len(data))
    training_set = shuffled_set[:]
    shuffled_set_test = random.sample(data_test, len(data_test))
    testing_set = shuffled_set_test[:]

    for el in training_set:
        train_set.append(el[1])
        zeros = [0] * classes
        zeros[int(el[0]) - 1] = 1
        train_labels.append(zeros)

    for el in testing_set:
        test_set.append(el[1])
        zeros = [0] * classes
        zeros[int(el[0]) - 1] = 1
        test_labels.append(zeros)

    train_token_ids = tokenizer(train_set, max_seq_length)
    test_token_ids = tokenizer(test_set, max_seq_length)

    train_labels_final = np.array(train_labels)
    test_labels_final = np.array(test_labels)

    return train_token_ids, train_labels_final, test_token_ids, test_labels_final

# Creazione del tokenizer
def createTokenizer():
    vocab_file = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12/vocab.txt")
    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=False)
    return tokenizer

# Creazione del modello BERT layer
def createBertLayer():
    global bert_layer
    bert_params = bert.params_from_pretrained_ckpt(modelBertDir)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    bert_layer.apply_adapter_freeze()

# Caricamento dei pesi pre-addestrati per il modello BERT
def loadBertCheckpoint():
    modelsFolder = os.path.join(modelBertDir, "multi_cased_L-12_H-768_A-12")
    checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")
    bert.load_stock_weights(bert_layer, checkpointName)

# Creazione del modello
def createModel():
    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
    ])
    model.build(input_shape=(None, max_seq_length))
    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    print(model.summary())

# Addestramento del modello
def fitModel(training_set, training_label, testing_set, testing_label):
    checkpointName = os.path.join(modelDir, "bert_faq.ckpt")
    # Callback per il salvataggio dei pesi del modello
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointName, save_weights_only=True, verbose=1)
    history = model.fit(
        training_set,
        training_label,
        epochs=300,
        validation_data=(testing_set, testing_label),
        verbose=1,
        callbacks=[cp_callback]
    )

# Caricamento del tokenizer
tokenizer = createTokenizer()

# Creazione del BERT layer
createBertLayer()

# Caricamento dei pesi pre-addestrati per BERT
loadBertCheckpoint()

# Creazione del modello
createModel()

# Caricamento dei dati e addestramento del modello
train_set, train_labels, test_set, test_labels = loadData(tokenizer)
fitModel(train_set, train_labels, test_set, test_labels)
