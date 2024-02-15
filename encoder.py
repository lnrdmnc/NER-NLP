import tensorflow.keras as keras
from tensorflow.keras import layers, Model
import tensorflow as tf



class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),  # Prima parte della Feed Forward Network: ReLU activation
            layers.Dense(embed_dim)  # Seconda parte della Feed Forward Network
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Esempio di utilizzo in un modello
def build_model(input_shape, embed_dim, ff_dim, num_heads, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = TransformerEncoderBlock(embed_dim=embed_dim, ff_dim=ff_dim, num_heads=num_heads)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)

# Parametri del modello
input_shape = (None, EMBED_DIM)  # Ad esempio, (sequence_length, embedding_dimension)
embed_dim = 256  # Dimensione dell'embedding
ff_dim = 512  # Dimensione della Feed Forward Network interna
num_heads = 8  # Numero di teste nell'attenzione multi-testa
num_classes = 10  # Numero di classi di output (ad esempio, per la classificazione)

model = build_model(input_shape, embed_dim, ff_dim, num_heads, num_classes)
model.summary()
