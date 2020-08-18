import tensorflow as tf
import matplotlib.pyplot as plt

def spectrogram(
    audio_path,
    channels=1,
    frame_length=256,
    frame_step=256,
    fft_length=None,
    brightness=100.0,
    plotable=True):
    """Decode and build a spectrogram from a wav file.

    Args:
      audio_path: String path to wave file.
      channels: Audio channel count.
      frame_length: An integer scalar Tensor. The window length in samples.
      frame_step: An integer scalar Tensor. The number of samples to step.
      fft_length: An integer scalar Tensor. The size of the FFT to apply. If not provided, uses the smallest power of 2 enclosing frame_length.
      brightness: Brightness of the spectrogram.
      plotble: Wether to return a plotable tensor or the original complex result
      
    Returns:
      2-D uint8 Tensor with the image contents.
    """
    
    brightness = tf.constant(brightness)

    # load file as string tensor
    input_file = tf.io.read_file(audio_path)
    # decode string tensor into "audio"(float-32 tensor)
    audio, sample_rate = tf.audio.decode_wav(input_file)

    # check number of channels
    if(channels != audio.shape[-1]):
        if(channels > audio.shape[-1]):
            raise Exception("CHANNEL ERROR: 'channels' must be <= number of channels in the file")
        else:
            print("WARNING: using only the first {} chanenls in the file".format(channels))
            

    stft = 0 # declare stft variable
    for channel in range(channels):
        # compute the short-time fourier transform and accumulate
        stft = stft + tf.signal.stft(tf.transpose(audio[:,channel]), frame_length=frame_length, frame_step=frame_step, fft_length=fft_length, pad_end=True)
    # average out the results
    stft = stft / channels
    

    if(plotable):
        return to_plotable(stft, brightness)
    else:
        return stft

def to_plotable(stft, brightness=100.0):
    
    # adjust brightness and normalize pixels
    mul = tf.multiply(tf.math.abs(stft), brightness)
    min_const = tf.constant(255.0)
    minimum = tf.minimum(mul, min_const)

    #resize = tf.image.resize(edims, [stft0.shape[0], stft0.shape[1]])
    #squeeze = tf.squeeze(resize, -1)

    # Tensorflow spectrogram has time along y axis and frequencies along x axis
    # so we fix that
    # expand dims to get proper shape
    """edims = tf.expand_dims(minimum, -1)
    flip_left_right = tf.image.flip_left_right(edims)
    transposed = tf.image.transpose(flip_left_right)
    squeeze = tf.squeeze(transposed, -1)"""

    # Cast to uint8 and encode as png
    cast = tf.cast(minimum, tf.uint8)
    out = tf.cast(cast, tf.float32) / 255.0

    #img = tf.image.encode_png(tf.cast(transposed, tf.uint8))
    #tf.io.write_file("spectrogram.png", img)

    return out #tf.expand_dims(out, -1)


#path_to_file = "C:/Users/Jose/Documents/Python/Source Separation/DSD100subset/Mixtures/Dev/055 - Angels In Amplifiers - I'm Alright/mixture_2250.wav"

#cast = spectrogram(path_to_file, 2, 256, 256, 256)
#plt.imshow(cast)
#plt.show()
