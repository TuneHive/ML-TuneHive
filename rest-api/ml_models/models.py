import keras
from keras.api.layers import Embedding, Layer, GRU, Attention, Concatenate, Dense, LeakyReLU, BatchNormalization, Dropout
from keras.api.models import load_model as load_keras_model, Model
import tensorflow as tf
import joblib
from utils.processor import remove_duplicates_with_logit_check
from .preprocessing import process_sequences

@keras.saving.register_keras_serializable(package="gru4rec_with_attention")
class ItemEmbedding(Layer):
    def __init__(self, num_items, item_embed_dim, **kwargs):
        super(ItemEmbedding, self).__init__(**kwargs)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=item_embed_dim, mask_zero=True)

    def call(self, items):
        # Embed items
        items_embedded = self.item_embedding(items)
        return items_embedded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_items": self.num_items,
            "item_embed_dim": self.item_embed_dim
        })
        return config

@keras.saving.register_keras_serializable(package="gru4rec_with_attention")
class GRU4REC(Model):
    def __init__(self, rnn_params, genre_embed_dim, item_embed_dim, ffn1_units, feature_dense_units,  items_size, genres_size, *args, **kwargs):
        super(GRU4REC, self).__init__(*args, **kwargs)

        self.rnn_params = rnn_params
        self.genre_embed_dim = genre_embed_dim
        self.item_embed_dim = item_embed_dim
        self.ffn1_units = ffn1_units
        self.feature_dense_units = feature_dense_units
        self.items_size = items_size
        self.genres_size = genres_size
        self.input_shape = None

        print(f"items size: {items_size}")
        print(f"genres size: {genres_size}")
        
        self.embedding = Embedding(input_dim=items_size, output_dim=item_embed_dim, mask_zero=True)
        
        # Genre embedding (only for genre, which is categorical and a string)
        self.genre_embedding = Embedding(input_dim=genres_size, output_dim=genre_embed_dim, mask_zero=True, name='genre_embedding')

        # RNN layers
        self.rnn_layers = []
        self.rnn_layers.append(GRU(**rnn_params[0], return_sequences=True))
        for i in range(1, len(rnn_params) - 1):
            self.rnn_layers.append(GRU(**rnn_params[i], return_sequences=True))
        self.rnn_layers.append(GRU(**rnn_params[-1], return_sequences=True))

        self.concat = Concatenate(axis=-1, name='concat_1')
        self.batch_norm = BatchNormalization(name='batchnorm')

        # Dropout layer
        self.dropout = Dropout(0.2, name='dropout')

        # Feed-forward layers
        self.feature_dense = Dense(feature_dense_units, activation='relu', name='feature_dense')  # Dense layer for features (if required)
        self.ffn1 = Dense(ffn1_units, name='ffn_1')
        self.activation1 = LeakyReLU(negative_slope=0.2, name='freaky_relu')
        self.item_output = Dense(items_size, name='item_output')

        # self.genre_output = Dense(preprocessed_data.genres_size, activation='softmax', name='genre_output')
        self.attention = Attention(use_scale=False, dropout=0.2, name='attention')

    def call(self, inputs, training=False):
        """
        Forward pass for the GRU4REC model.
        :param inputs: Tuple (item_sequences, item_features, item_genres)
        :param training: Boolean indicating if the model is in training mode
        """
        item_sequences, _ , item_genres = inputs
        
        # Update input shape dynamically
        if self.input_shape is None:
            # Set the input shape based on the first batch of inputs
            self.input_shape = item_sequences.shape
        encoding_padding_mask = tf.math.logical_not(tf.math.equal(item_sequences, 0))

        # print("Item Sequence Shape:", item_sequences.shape)
        # print("Item Genres Shape:", item_genres.shape)
        
        # Embed items
        item_embedded = self.embedding(item_sequences)
        item_embedded = tf.expand_dims(item_embedded, axis=2)

        # Genre embedding
        genre_embedded = self.genre_embedding(item_genres)

        # print("Item Embedded Shape:", item_embedded.shape)
        # print("Genre Embedded Shape:", genre_embedded.shape)
        genre_embedded = tf.reduce_mean(genre_embedded, axis=2)
        genre_embedded = tf.expand_dims(genre_embedded, axis=2)

        # Feature transformation (features are passed directly as floats, so no embedding is needed)
        # feature_transformed = self.feature_dense(item_features)
        # feature_transformed = tf.expand_dims(feature_transformed, axis=1)

        # combined_input = tf.concat([item_embedded, feature_transformed, genre_embedded], axis=-1)
        combined_input = tf.concat([item_embedded, genre_embedded], axis=-1)
        combined_input = self.batch_norm(combined_input)
        # print("Combined input shape:",combined_input.shape)

        # Pass through RNN layers
        combined_input = tf.reduce_mean(combined_input, axis=-2)
        # print("Reduced input shape:", combined_input.shape)

        x = combined_input
        x = self.rnn_layers[0](x)
        x = self.dropout(x, training=training)

        for i in range(1, len(self.rnn_layers)):
            x = self.concat([combined_input, x])  # Concatenate item embeddings with RNN outputs
            x = self.rnn_layers[i](x)
            x = self.dropout(x, training=training)
        # x = self.batch_norm(x)
        
        # # Give attention
        # encodding_padding_mask = tf.expand_dims(encoding_padding_mask, axis=1)

        # # x = tf.expand_dims(x, axis=1)
        # print(f"Shape before attention: {x.shape}")
        # x = self.attention(inputs=[x,x], mask=[encodding_padding_mask, encodding_padding_mask], use_causal_mask=True)

        # Feed-forward layers
        # print(f"Shape before squeeze: {x.shape}")
        # x = tf.squeeze(x, axis=1)
        # print(f"Shape before softmax: {x.shape}")

        x = self.ffn1(x)
        x = self.dropout(x, training=training)
        x = self.activation1(x)

        # print(f"Shape after activation: {x.shape}")
        item_logits = self.item_output(x)  # Item prediction
        # print(f"Output shape: {item_logits.shape}")

        return item_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "rnn_params": self.rnn_params,
            "genre_embed_dim": self.genre_embed_dim,
            "item_embed_dim": self.item_embed_dim,
            "ffn1_units": self.ffn1_units,
            "feature_dense_units": self.feature_dense_units,
            "items_size": self.items_size,
            "genres_size": self.genres_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def predict(model, item_sequence, genres_sequence, item_length):
    padded_items_sequence, padded_genres_sequence, features_sequence = process_sequences(item_sequence, genres_sequence, 15)
    sequence = (padded_items_sequence, features_sequence,padded_genres_sequence)
    predicted_logits_sequence = model(sequence, training=False)
    predicted_sequence = remove_duplicates_with_logit_check(predicted_logits_sequence, item_length)
    return predicted_sequence

def load_encoder(encoder_path: str):
    loaded_encoder = joblib.load(encoder_path)
    return loaded_encoder

def load_model(model_path: str):
    loaded_model = load_keras_model(
        model_path,
        safe_mode=False,
        custom_objects = {
            'GRU4REC': GRU4REC
        }
    )
    return loaded_model
    