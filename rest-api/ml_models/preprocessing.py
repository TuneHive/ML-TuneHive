import tensorflow as tf

import time
from sklearn.preprocessing import LabelEncoder
from keras.api.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import ast
import numpy as np



# Feature columns (as provided)

feature_columns = [
    'spotify_genre',
]

# Define the DataPreprocessor class
class DataPreprocessor:
    def __init__(self, df, feature_columns, batch_size=16, fixed_genre_size=15, train_size=0.8):
        """
        Initializes the data preprocessor with necessary parameters and preprocessing layers.
        Args:
            df (DataFrame): The input DataFrame containing session data.
            feature_columns (list): List of feature column names.
            batch_size (int): The batch size for dataset creation.
            fixed_genre_size (int): The fixed size for genre vectorization.
            train_size (float): Proportion of the data to use for training (between 0 and 1).
        """
        self.df = df
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.fixed_genre_size = fixed_genre_size
        self.train_size = train_size

        # Split the dataset into training and testing datasets
        self.train_df, self.test_df = train_test_split(self.df, train_size=self.train_size, random_state=42)
        
        # Numeric feature preprocessing
        self.numeric_data = self.df[feature_columns[1:]].apply(pd.to_numeric, errors='coerce')
        self.mean_values = self.numeric_data.mean()
        self.std_values = self.numeric_data.std()

        # Initialize LabelEncoder for SongID and spotify_genre
        self.song_id_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()

        # Extract unique SongIDs and genres
        unique_song_ids = self.df['SongID'].unique()
        all_genres = []
        for genre_str in self.df['spotify_genre']:
            try:
                genre_list = ast.literal_eval(genre_str)  # Safely parse the string into a list
                if isinstance(genre_list, list):
                    all_genres.extend(genre_list)
            except Exception as e:
                print(f"Error parsing genre: {e}")

        unique_genres = list(set(all_genres))

        # Fit the LabelEncoders on the data
        self.song_id_encoder.fit(unique_song_ids)
        self.genre_encoder.fit(unique_genres)

        self.items_size = len(self.song_id_encoder.classes_)  # Number of unique SongIDs
        self.genres_size = len(self.genre_encoder.classes_)

        self.dataset = None

    def preprocess_song_id(self, song_id):
        """
        Encode the SongID using LabelEncoder.
        """
        return self.song_id_encoder.transform([song_id])[0]

    def clean_genre(self, value, default_value=0, dtype=tf.int32):
        """
        Clean and process the 'spotify_genre' feature.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return np.full((self.fixed_genre_size,), default_value, dtype=dtype.as_numpy_dtype)
        try:
            genre_list = eval(value) if isinstance(value, str) else value
            if isinstance(genre_list, list):
                genre_encoded = self.genre_encoder.transform(genre_list)
            else:
                genre_encoded = self.genre_encoder.transform([value])
        except Exception:
            genre_encoded = self.genre_encoder.transform([value])

        # Pad or truncate to fixed size
        return np.pad(genre_encoded, (0, max(0, self.fixed_genre_size - len(genre_encoded))),
                      mode='constant')[:self.fixed_genre_size].astype(dtype.as_numpy_dtype)

    def clean_numeric_feature(self, value, default_value=0.0, feature_name="feature", mean=None, std=None):
        """
        Clean, process, and normalize numerical features using Z-score normalization.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default_value
        try:
            value = float(value)
            # Apply Z-score normalization if mean and std are provided
            if mean is not None and std is not None and std != 0:
                z_score_value = (value - mean) / std
                return z_score_value
            return value  # Return raw value if no normalization

        except ValueError:
            return default_value

    def create_session_dataset(self, session_df):
        """
        Create session dataset as a list of dictionaries for each session.
        """
        session_df = session_df.sort_values(by=['session_id', 'TimeStamp_UTC'])
        grouped = session_df.groupby('session_id')
        sessions_data = []
        for session_id, group in grouped:
            session_data = group.to_dict(orient='records')
            sessions_data.append(session_data)
        return sessions_data

    def preprocess_data(self, sessions, k=1):
        """
        Preprocess session data into TensorFlow dataset with split genre and features,
        filtering out sequences where the next item sequence length is not greater than 10.
        """
        item_sequences = []
        next_item_sequences = []
        genre_sequences = []
        next_genre_sequences = []
        feature_sequences = []
        processed_item_count = 0

        for idx, session in enumerate(sessions):
            # Filter the session that has length less than k
            if len(session) < k:
                continue
            session_item_sequences = []
            session_next_item_sequences = []
            session_genre_sequences = []
            session_next_genre_sequences= []
            session_feature_sequences = []
            session_id = session[0]['session_id']

            for i in range(len(session) - 1):
                # Process items
                session_item_encoded = self.preprocess_song_id(session[i]['SongID'])
                next_session_item_encoded = self.preprocess_song_id(session[i + 1]['SongID'])
                session_item_sequences.append(session_item_encoded)
                session_next_item_sequences.append(next_session_item_encoded)

                # Process genre
                genre_cleaned = self.clean_genre(session[i].get('spotify_genre', None))
                next_genre_cleaned = self.clean_genre(session[i+1].get('spotify_genre', None))
                session_genre_sequences.append(genre_cleaned)
                session_next_genre_sequences.append(next_genre_cleaned)

                # Process numerical features
                numeric_features = []
                for col in self.feature_columns:
                    if col != 'spotify_genre':
                        mean = self.mean_values.get(col, None)
                        std = self.std_values.get(col, None)
                        cleaned_feature = self.clean_numeric_feature(session[i].get(col, None), mean=mean, std=std)
                        numeric_features.append(cleaned_feature)
                session_feature_sequences.append(numeric_features)

            # Filter out sessions where the next item sequence length is not greater than 10
            # Extend sequences only if the next item sequence length is greater than 10
            print("session item sequences:", session_item_sequences)
            print("session next item sequences:", session_next_item_sequences)

            # Filter the item that have session length less than k
            item_sequences.append(session_item_sequences)
            next_item_sequences.append(session_next_item_sequences)
            genre_sequences.append(session_genre_sequences)
            next_genre_sequences.append(session_next_genre_sequences)
            feature_sequences.append(session_feature_sequences)
            processed_item_count += len(session_item_sequences)
            #     # print(f"Session {idx + 1} processed with {len(session_item_sequences)} items.")
            # else:
            #     print(f"Session {idx + 1} skipped because next item sequence length is {len(session_next_item_sequences)}.")
        print(f"Total processed items: {processed_item_count}")

        # Pad sequences
        item_sequences = pad_sequences(item_sequences, padding='pre', value=0)
        next_item_sequences = pad_sequences(next_item_sequences, padding='pre', value=0)
        genre_sequences = pad_sequences(genre_sequences, padding='pre', value=0)
        next_genre_sequences = pad_sequences(next_genre_sequences, padding='pre', value=0)
        feature_sequences = pad_sequences(feature_sequences, padding='pre', dtype='float32', value=0.0)
        # print(f"item_sequences shape: {item_sequences.shape}")
        # print(f"next_item_sequences shape: {next_item_sequences.shape}")
        # print(f"genre_sequences shape: {genre_sequences.shape}")
        # print(f"next_genre_sequences shape: {next_genre_sequences.shape}")
        # print(f"feature_sequences shape: {feature_sequences.shape}")
        # print("item sequence padded:", item_sequences)
        # print("next item sequence padded:", next_item_sequences)
        # print("genre sequence padded:", genre_sequences)
        # print("next genre sequence padded:", next_genre_sequences)
        # print("feature sequence padded:", feature_sequences)

        # Create TensorFlow dataset
        sequence_length = item_sequences.shape[1]  # Assuming all sequences have the same length after padding
        print(f"Sequence length after padding: {sequence_length}")

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'item': item_sequences,
            'genre': genre_sequences,
            'features': feature_sequences,
            'next_item': next_item_sequences,
            'next_genre': next_genre_sequences
        })
    
        return dataset, sequence_length

    def create_session_dataset_tensor(self, k=1):
        """
        Main function to create session dataset as tensors and return the dataset.
        """
        if self.dataset is not None:
            print("Dataset already created")
            return

        print("Creating session dataset")
        sessions_data = self.create_session_dataset(self.df)
        dataset, sequence_length = self.preprocess_data(sessions_data, k=k)

        # Shuffle and batch the training data
        dataset = (
            dataset.batch(self.batch_size, drop_remainder=True)
                   .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        self.dataset = dataset
        return dataset, sequence_length

    def create_train_dataset(self, k=1):
        """
        Main function to create session dataset as tensors and return the dataset.
        """
        if self.dataset is not None:
            print("Dataset already created")
            return

        print("Creating session dataset")
        sessions_data = self.create_session_dataset(self.train_df)  # Use train data for training
        print("Creating tensor dataset")
        dataset = self.preprocess_data(sessions_data, k=k)

        # Shuffle and batch the training data
        dataset = (
            dataset.shuffle(buffer_size=1024)
                   .batch(self.batch_size, drop_remainder=True)
                   .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

    def get_test_data(self, k):
        """
        Return preprocessed test dataset without shuffling.
        """
        sessions_data = self.create_session_dataset(self.test_df)
        dataset = self.preprocess_data(sessions_data, k)
        
        # Batch the test data without shuffling
        dataset = (
            dataset.batch(self.batch_size, drop_remainder=True)
                   .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

    def batch_timer(self, dataset):
        """
        Timer function to track the time taken for batch processing.
        """
        for batch in dataset:
            start_time = time.time()
            # Simulate processing (e.g., model training or data transformation)
            end_time = time.time()
            batch_time = end_time - start_time
            print(f"Batch processing time: {batch_time:.4f} seconds")

def process_sequences(items_sequence, genres_sequence, target_sequence_length, extra_dimension=15):
    # Step 1: Pad items_sequence to target length
    padded_items_sequence = np.pad(
        items_sequence, 
        (target_sequence_length - len(items_sequence), 0), 
        mode='constant', 
        constant_values=0
    )

    # Step 2: Process genres_sequence
    # Initialize an array of zeros with the shape (target_sequence_length, extra_dimension)
    padded_genres_sequence = np.zeros((target_sequence_length, extra_dimension))

    for i in range(target_sequence_length):
        if i < len(genres_sequence):
            # Pad or truncate each genre list to `extra_dimension`
            genre_row = genres_sequence[i]
            padded_row = np.pad(
                genre_row, 
                (0, max(0, extra_dimension - len(genre_row))),  # Pad to the right
                mode='constant', 
                constant_values=0
            )[:extra_dimension]  # Ensure truncation if the list is longer than `extra_dimension`
        else:
            # Beyond the original sequence length, keep zeros
            padded_row = np.zeros(extra_dimension)

        # Assign the processed row to the padded_genres_sequence
        padded_genres_sequence[i] = padded_row

    # Step 3: Add batch dimension for both sequences
    padded_items_sequence = np.expand_dims(padded_items_sequence, axis=0)  # Shape: (1, sequence_length)
    padded_genres_sequence = np.expand_dims(padded_genres_sequence, axis=0)  # Shape: (1, sequence_length, num_features)

    # Step 4: Handle features_sequence (if any)
    features_sequence = np.array([])  # Placeholder
    features_sequence = np.expand_dims(features_sequence, axis=0) if features_sequence.size > 0 else np.empty((1, 0))

    # Return the padded sequences
    return padded_items_sequence, padded_genres_sequence, features_sequence