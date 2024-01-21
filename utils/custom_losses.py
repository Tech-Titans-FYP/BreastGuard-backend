from keras import backend as K
import tensorflow as tf

# Define your custom losses and metrics here
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def weighted_custom_loss(weight_factor):
    def custom_loss(y_true, y_pred):
        # Counting the number of white pixels (tumor region) in the true mask
        tumor_size = tf.reduce_sum(tf.cast(tf.equal(y_true, 255), tf.float32))

        # Calculate weights based on tumor size - larger tumors get higher weights
        weights = 1 + weight_factor * tumor_size

        # Standard binary cross-entropy loss
        binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

        # Apply weights to the loss
        weighted_loss = weights * binary_cross_entropy_loss

        # Return the mean loss
        return tf.reduce_mean(weighted_loss)
    return custom_loss

# Assuming the weight_factor was 0.5 when the model was trained
weight_factor = 0.5
custom_loss_fn = weighted_custom_loss(weight_factor)