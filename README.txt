This is the training system for Team Volgemobile's 2022-2023 Capstone

Task:
Construct system that autonomously flags target audio input patterns.

Approach:
Discretize audio input and transform it into a log mel spectrogram graph/image.

Fine tune a pre-trained image recognition neural network to classify the presence of target audio signals in a spectrogram.

Generate a flag every time a target signal is detected in the audio input.
_________

Spectrogram factory constructs ( 8400 clips * 4 second splits) = 36,000 1 second spectrograms of labeled audio using torch audio library

	spectrogram_factory_torchAudio-v1.5-1second_split.ipynb

Model Trainer builds an image data generator for keras model training.  It then loads a pre-trained model, removes the classification head, and re-trains/fine-tunes it using our new dataset.
	
	Model_Trainer_Try11-sigmoid-categ-accuracy.ipynb

Audio classifier runs a model produced by model_trainer live.  It processes audio input from a live mic or headset in 1 second chunks and prints the spectrogram generated and model classification for each second of audio input (up to 5 seconds).

	audio_classifier-v1.34

base_model_3_8_2023_sigmoid-try11-3 is the final model for 2022