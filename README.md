# sound-cnn
A convolutional neural network that classifies sounds. This is accomplished by taking raw audio information and converting it into spectogram information. Each spectogram is a "picture" of the sound, which the CNN learns to classify in the same way that traditional image recognition paradigms work.

For more info on how this was accomplished, and how it compares to other methods, read the following [Medium post](https://medium.com/@awjuliani/recognizing-sounds-a-deep-learning-case-study-1bc37444d44d#.5ubhfdh0h).

###Setup
`pip install -r requirements.txt`

### Training
The model can be trained with the following arguments:

`$ python train.py 'bpm' 'sampling rate' 'audio path' 'iterations' 'batch size'`

`bpm` is dependent on the sound files being classified

`sampling rate` is most often set to 44100.

`audio path` is the directory where the audio files are located. The program will read each file in the directory as a separate sound class, for example: if the directory has
two files file1.wav and file2.wav, then there will be two classes that the CNN will attempt to learn to identify.

`iterations` should vary depending on the difficulty of the classification. 1000 ~ 5000 may be ideal for most situations.

`batch size` is most often set to 100 ~ 200.

####Example
`python train.py 240 44100 audio/ 1000 150`
