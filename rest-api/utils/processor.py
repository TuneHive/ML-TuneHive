import tensorflow as tf

def remove_duplicates_with_logit_check(logits, sequence_length):
    # Initialize an empty list for the final sequence
    predicted_sequence = []
    print(logits.shape)
    if (sequence_length > logits.shape[1]):
        sequence_length = logits.shape[1]
    # Iterate through the logits (probabilities for each token)
    for step in range(sequence_length):
        # Get the current logits and their indices (i.e., token indices)
        current_logits = logits[0][step]
        
        # Sort the logits in descending order to find the highest probability
        top_indices = tf.argsort(current_logits, direction='DESCENDING')  # Sort logits
        
        # Find the next highest logit that is not a duplicate
        for idx in top_indices:
            if idx.numpy() not in predicted_sequence:  # Check if the token is already in the sequence
                predicted_sequence.append(idx.numpy())  # Add the token to the sequence
                break  # Stop once a unique token is found
    
    return predicted_sequence