# Audio Separator

this project contains a Deep Learning model for audio source separation.
The program takes an audio file as input and will output up to 4 audio files containing:
* Vocals
* Accompaniment
* Bass
* Drums

## Pre-procesing

Before feeding the data into the model, an spectrogram is generated from the audio input, since the frequency domain data is much more representative of the different audio sources than the combined time domain data

## The DL Model

To achieve the best quality posible, a model is trained separately to isolate each of the audio sources and they are then combined into a single bigger model.
All four models are identical in structure, and trained using the same parameters and dataset, they only differ in their target source.

## The Dataset

For this project the MUSDB18 dataset was used, this data set consists of a total of 150 full-track songs of different styles and includes both the stereo mixtures and the original sources, divided between a training subset and a test subset.

The trainig set is composed of 100 songs, and the test set is composed of 50 songs.

All files from the musdb18 dataset are encoded in the Native Instruments stems format (.mp4). It is a multitrack format composed of 5 stereo streams, each one encoded in AAC @256kbps. These signals correspond to:

* 0 - The mixture,
* 1 - The drums,
* 2 - The bass,
* 3 - The rest of the accompaniment,
* 4 - The vocals.

For each file, the mixture correspond to the sum of all the signals. All signals are stereophonic and encoded at 44.1kHz.
