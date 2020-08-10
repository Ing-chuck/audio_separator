import tensorflow as tf
from tensorflow.data import Dataset
import matplotlib.pyplot as plt

from spectrogram import spectrogram, to_plotable
from generate_dataset import generate_dataset
from model import create_model

# spectrogram parameters
channels = 2
window_size = 1024
frame_step = 256

# dataset grneration parameters
mask_threshold = 0.8
context_frames = 50 # ~300ms of context per prediction

# Training parameters
epochs = 10
batch_size = 64

path_to_files = ["C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/055 - Angels In Amplifiers - I'm Alright/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/081 - Patrick Talbot - Set Me Free/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Test/005 - Angela Thomas Wade - Milk Cow Blues/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Test/049 - Young Griffo - Facade/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/055 - Angels In Amplifiers - I'm Alright/bass.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/055 - Angels In Amplifiers - I'm Alright/drums.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/055 - Angels In Amplifiers - I'm Alright/other.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/055 - Angels In Amplifiers - I'm Alright/vocals.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/bass.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/drums.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/other.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/vocals.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/bass.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/drums.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/other.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/vocals.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/049 - Young Griffo - Facade/bass.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/049 - Young Griffo - Facade/drums.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/049 - Young Griffo - Facade/other.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/049 - Young Griffo - Facade/vocals.wav"]

# files for vocal training
training_files = ["C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/055 - Angels In Amplifiers - I'm Alright/mixture.wav",
                  "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/055 - Angels In Amplifiers - I'm Alright/vocals.wav",
                  "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/081 - Patrick Talbot - Set Me Free/mixture.wav",
                  "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/vocals.wav"]

testing_files = ["C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Test/005 - Angela Thomas Wade - Milk Cow Blues/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/vocals.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Test/049 - Young Griffo - Facade/mixture.wav",
                 "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Test/049 - Young Griffo - Facade/vocals.wav"]

# load files and calculate spetrograms
print("loading files")
training_spectrograms = []
for path in training_files:
    training_spectrograms.append(spectrogram(path, channels, window_size, frame_step, plotable=False))

testing_spectrograms = []
for path in testing_files:
    testing_spectrograms.append(spectrogram(path, channels, window_size, frame_step, plotable=False))

# generate training and testing datasets
# FIXME:
# there should be a shorter way to do this
training_features = []   # will hold the training data set
training_labels = []
examples = []   # total number of examples
for i in range(0, len(training_spectrograms), 2):
    mix = training_spectrograms[i]
    vocals = training_spectrograms[i+1]

    # Calculate mask
    mask = (tf.math.abs(vocals) / tf.math.abs(mix))
    # convert to binary, taking all the values grater than threshold
    mask = tf.math.greater_equal(mask, mask_threshold)
    # cast as float to feed into loss function
    mask = tf.cast(mask, tf.float32)

    # transpose the spectrograms for dataset generation
    mask = tf.transpose(mask)
    mix = tf.transpose(mix)

    mix = tf.math.abs(mix)
    mix = to_plotable(mix)
    
    ds, n = generate_dataset(mix, mask, context_frames)
    training_features.extend(ds[0])
    training_labels.extend(ds[1])
    examples.append(n)

# get input shape
input_shape = training_features[0].shape.as_list()[1:]
print(input_shape)
"""
training_set = training_sets[0]
for i in range(1, len(training_sets)):
    training_set[0].extend(training_sets[i][0][i])
    training_set[1].extend(training_sets[i][1][i])
    print("{} - {}".format(
    #tf.concat([training_set[0], training_sets[i][0]], 0)
    #tf.concat([training_set[1], training_sets[i][1]], 0)
"""
training_set = Dataset.from_tensor_slices((training_features, training_labels))

test_features = []   # will hold the training data set
test_labels = []
tests = []
for i in range(0, len(testing_spectrograms), 2):
    mix = testing_spectrograms[i]
    vocals = testing_spectrograms[i+1]
    # Calculate mask
    mask = (tf.math.abs(vocals) / tf.math.abs(mix))
    # convert to binary, taking all the values grater than threshold
    mask = tf.math.greater_equal(mask, mask_threshold)
    # cast as float to feed into loss function
    mask = tf.cast(mask, tf.float32)

    # transpose the spectrograms for dataset generation
    mask = tf.transpose(mask)
    mix = tf.transpose(mix)

    mix = tf.math.abs(mix)
    mix = to_plotable(mix)
    
    ds, n = generate_dataset(mix, mask, context_frames)
    test_features.extend(ds[0])
    test_labels.extend(ds[1])
    tests.append(n)

test_set = Dataset.from_tensor_slices((test_features, test_labels))

# unload unnecessary data
del training_spectrograms
del testing_spectrograms
del training_features
del training_labels
del test_features
del test_labels

model = create_model(input_shape)

# prepare datasets for training
train_batches = training_set.repeat(epochs).shuffle(sum(examples), reshuffle_each_iteration=True).prefetch(batch_size)
test_batches = test_set.repeat(epochs).shuffle(sum(tests), reshuffle_each_iteration=True).prefetch(batch_size)

# fit model to data
model.fit(train_batches, validation_data=test_batches, epochs=epochs, verbose=2)
model.save('ss_conv_2_layer_leakyRelu_sgd_bce.h5')
