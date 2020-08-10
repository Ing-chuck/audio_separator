import tensorflow as tf
import math

def generate_dataset(
    feature_spec,
    label_spec,
    window_size
    ):
    """Create (feature, label) pairs from spectrogram tensors.
    feature_spec: feature spectrogram tensor.
    label_spec: label spectrogram tensor.
    window_size: size of window to cover the data

    Returns:
        tf.data.Dataset object containig feature, label pairs.
    """

    # check data shape
    if(feature_spec.shape != label_spec.shape):
        raise Exception("VALUE_ERROR: feature_spec and label_spec must have the same shape")

    frames = int(feature_spec.shape[-1])
    middle_frame = math.ceil(window_size / 2) - 1
    delta = math.floor(window_size / 2)
    if(window_size % 2 == 0): # window_size is even
        delta -= 1

    x = []  # will hold the feature tensors
    y = []  # will hold the label tensors
    # move the window over the spectrogram one frame at a time
    n = 0   # holds number of elements
    for frame in range(middle_frame, frames):
        start = frame - delta
        end = start + window_size
        
        if(end > frames):
            break
        
        #x.append(tf.expand_dims(tf.expand_dims(feature_spec[:,start:end], -1), 0))
        x.append(tf.expand_dims(feature_spec[:,start:end], -1))
        y.append(tf.expand_dims(label_spec[:,frame], 0))
        n += 1

    #print(tf.convert_to_tensor(x))
    #print(tf.convert_to_tensor(y))
        
    # dimension sanity check
    for i, xi in enumerate(x):
        if(len(xi.shape.as_list()) < 4):
            x[i] = tf.expand_dims(xi, 0)

    for yi, i in enumerate(y):
        if(len(xi.shape.as_list()) < 2):
            y[i] = tf.expand_dims(yi, 0)
    
    return (x,y), n
