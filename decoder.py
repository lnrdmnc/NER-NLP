from keras import layers, Model
import tensorflow as tf
from model.PositionalEmbedding import PositionalEmbedding

class TransformerDecoderBlockForNER(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, num_labels, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_labels = num_labels  # Numero di etichette di NER uniche nel dataset

        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        # Nota: Assicurati che PositionalEmbedding sia configurato correttamente per il tuo caso d'uso
        self.embedding = PositionalEmbedding(embed_dim=embed_dim, sequence_length=None, vocab_size=None)

        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.1)

        # Cambiato da Dense VOCAB_SIZE a num_labels per la classificazione NER
        self.classifier = layers.Dense(num_labels, activation="softmax")  # Output layer per la classificazione NER

    def call(self, inputs, training=False, mask=None):
        inputs = self.embedding(inputs)

        # Prima attenzione multi-testa (self-attention)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=None, training=training)
        attention_output_1 = self.dropout_1(attention_output_1, training=training)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Feedforward network
        ffn_output = self.ffn_layer_1(out_1)
        ffn_output = self.ffn_layer_2(ffn_output)
        ffn_output = self.dropout_2(ffn_output, training=training)
        ffn_output = self.layernorm_2(out_1 + ffn_output)

        # Classificazione delle entità
        predictions = self.classifier(ffn_output)

        return predictions

# Esempio di utilizzo
# Definizione delle dimensioni
EMBED_DIM = 512
FF_DIM = 2048
NUM_HEADS = 8
NUM_LABELS = 9  # Ad esempio, per un dataset con 9 etichette di entità diverse

# Creazione del modello
decoder_block = TransformerDecoderBlockForNER(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, num_labels=NUM_LABELS)

# Input tensor
input_tokens = tf.keras.Input(shape=(None,), dtype=tf.int32)  # 'None' per lunghezza sequenza variabile

# Output del modello
predictions = decoder_block(input_tokens)

# Creazione del modello Keras
model = Model(inputs=input_tokens, outputs=predictions)

# Visualizzazione del modello
model.summary()
