
import tensorflow as tf

LABEL_KEY = "Tutoring"
FEATURE_KEY = "GPA"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    # Directly use the FEATURE_KEY as a float without conversion to string
    outputs[transformed_name(FEATURE_KEY)] = tf.cast(inputs[FEATURE_KEY], tf.float32)

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
