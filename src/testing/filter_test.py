import tensorflow as tf
from tensorflow.data import Dataset
import matplotlib.pyplot as plt

from spectrogram import spectrogram
from generate_dataset import generate_dataset

training_files = ["C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/081 - Patrick Talbot - Set Me Free/mixture.wav",
                  "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Sources/Dev/081 - Patrick Talbot - Set Me Free/vocals.wav"
                  ]

# load files and calculate spetrograms               
training_spectrograms = []
for path in training_files:
    training_spectrograms.append(spectrogram(path, 2, 1024, 256, plotable=False))

#for i in range(0, len(training_spectrograms), 2):
mix = training_spectrograms[0]
vocals = training_spectrograms[1]
# calculate mask
mask = (tf.math.abs(vocals) / tf.math.abs(mix))
# normalize
#mask = mask / tf.math.reduce_max(mask)
# convert to binary
mask = tf.math.greater_equal(mask, 0.8)


recovered = (mix * tf.cast(mask, tf.complex64))


wave = tf.signal.inverse_stft(
    recovered, 1024, 256,
    window_fn=tf.signal.inverse_stft_window_fn(256)
)
wave = tf.audio.encode_wav(tf.reshape(wave, (-1,1)), 44100)
tf.io.write_file("recovered_vocals.wav", wave)

"""
plt.subplot(311)
plt.imshow(wave)
plt.subplot(312)
plt.imshow(vocals)
plt.subplot(313)
plt.imshow(recovered)
plt.show()
"""
