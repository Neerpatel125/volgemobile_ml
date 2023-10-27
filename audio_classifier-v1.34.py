#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Audio Input
import pyaudio
# import time
import numpy as np

# Spectrogram Factory
# torch audio to do gpu spectrograms
# librosa to do regular spectrograms
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
#import librosa

# Neural Network
# Import inception model
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
import keras
# from keras.utils import load_img
from keras.utils import img_to_array

# Other libraries
import os
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib import cm # import plt colormap module
from PIL import Image # PIL required for RGBA image conversion
from matplotlib.gridspec import GridSpec # Gridspec required for better layout

import time


# In[2]:


# Please set default directory for saved model:

# The directory where the model is saved
model_save_path = r"N:\# GMU 2022 ML Model\UrbanSound8K\audio\trainset_second"

# The name of the model
base_model_name = r"base_model_3_8_2023_sigmoid-try11-2"


# In[3]:


# 1.34 moved 
# pip install torch===1.13.1+cu116 torchvision===0.13.1+cu116 torchaudio===0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html


# In[4]:


#  pip install --upgrade --quiet jupyter_client ipywidgets


# In[5]:


### Please install Torch and all Torch libraries using this command and ensure the version numbers match

# print(torch.__version__)
# print(torchaudio.__version__)
# # print(torchvision.__version__)


# In[6]:


# ### Please ensure your tf.__version__ is 2.10.0 and your keras.__version__ is also 2.10.0
# print(f"tf.__version__ is: {tf.__version__}")
# print(f"keras.__version__ is: {keras.__version__}")


# In[7]:


# CUDA Test
# geeksforgeeks
# https://www.geeksforgeeks.org/how-to-set-up-and-run-cuda-operations-in-pytorch/
import torch

import tensorflow as tf

from tensorflow.python.client import device_lib

def cuda_test():
    #
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    print("\n\n\n")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    
    #
    print("\n\n\n")
    print("Tensorflow GPU Test")
    num_devices = tf.config.list_physical_devices('GPU')

    print("Num GPUs Available: ", len(num_devices))
    print(device_lib.list_local_devices())
    
    #
    print("\n\n\n")
    if torch.cuda.is_available():
        print('We have a GPU!')
    else:
        print('Sorry, CPU only.')


# In[8]:


# https://forums.developer.nvidia.com/t/install-torchaudio-and-torchvison-from-wheel/230097
cuda_test()


# In[9]:


# Recording Function

# Warning, 
# Complete_Systemv1.2 encapsulation broken by accessing external pyaudio
# Complete_Systemv1.3 record_audio() now requires an open stream to be passed from main
# 
def record_audio(seconds_to_record=5, stream = None, num_chunks_to_record = None, CHUNK = None):
    # Moved out of the function
#     #Reference List of Parameters
#     # Load parameters
#     SRATE = 16000
#     # T_INTERVAL = 1/SRATE
#     CHUNK = 1024 # FRAMES_PER_BUFFER = CHUNK = normally 1024 
#     FORMAT = pyaudio.paInt16
#     NUM_CHANNELS = 1
#     # STRIDE = 2048
#     # VOLUME = .25 
    
#     # Instantiate PyAudio
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT,
#                     channels=NUM_CHANNELS,
#                     rate=SRATE,
#                     input=True,
#                     output=True,
#                     )

    chunk_count = 0
    input_data = np.array([])
    
    while chunk_count < num_chunks_to_record:
        #print('Time Elapsed: ' + str(time.time() - t_start))
        chunk_count += 1
        #t_start = time.time()
        # Read input from sensor
        input_data = np.append(input_data, stream.read(CHUNK, exception_on_overflow = False))

        # Append current input buffer to the raw binary history
        #     recording_history = b"".join([recording_history,input_data])

    formatted_input = np.frombuffer(input_data, dtype=np.int16)

    return formatted_input


# In[10]:


# Pads an input signal less than chunk_size_needed and truncates the ends of signals that are more than 
# one second but not exactly a multiple of 1 second.
def pad_or_shorten(signal, chunk_size_needed):
    signal_processed = signal
    # Pad if example is less than 1 second
    if (len(signal) < chunk_size_needed):
        num_missing_samples = chunk_size_needed - len(signal)%chunk_size_needed
        signal_processed = torch.nn.functional.pad(input = signal, pad = (0, num_missing_samples))
    elif (len(signal) > chunk_size_needed):
        signal_processed = signal[:(chunk_size_needed*(len(signal)//chunk_size_needed))]
        #print(f"Signal Loss: {len(signal_processed)-len(signal)}")
    
    return signal_processed


# In[11]:


# Create mel spectrograms of audio signals in the form of np.int16 ndarray

# Warning,
# Complete_Systemv1.34 create_mel_spectrogram() now requires a preconstructed torch_transformer as input

def create_mel_spectrogram(signal, sample_rate, torch_transformer):
    # Find audio files in target directory
    # Convert np pcm to torch tensor:
    # https://www.kaggle.com/code/fanbyprinciple/video-audio-tesnor-conversion-using-pytorch 
    # https://stackoverflow.com/questions/73787169/how-to-turn-a-numpy-array-mic-loopback-input-into-a-torchaudio-waveform-for-a
    audio_tensor = torch.tensor(signal).type(torch.FloatTensor)
        
    torch_transformer_tutorial = torch_transformer
    
    # Moved out of this function to main function
#     # Create torch transformer for mel spectrograms
#     # Parameters
#     frame_size = 512
#     hop_length = 256
#     sr = 16000

#     # Create torch transformer
#     torch_transformer_tutorial = T.MelSpectrogram(
#     sample_rate=torch_sr,
#     n_fft=torch_frame_size,
#     win_length=torch_frame_size,
#     hop_length=torch_hop_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     norm="slaney",
#     onesided=True,
#     n_mels=128,
#     mel_scale="htk")    


#     signal = mix_down_if_necessary(signal)
#     signal = resample_if_necessary(signal, sr_file, sr)
    # print(tf.shape(signal))

    audio_tensor = pad_or_shorten(audio_tensor, sample_rate)
    #print(f"Audio tensor shape after pad or shorten: {tf.shape(audio_tensor)}")
    # Split signal into 1 second parts
    signal_arr = torch.split(audio_tensor, 16000)
    
    
    #print(f"signal_arr shape after torch.split: {len(signal_arr)}")
    #print(f"signal_arr[0] shape after torch.split: {len(signal_arr[0])}")
    #print(f"signal_arr[0].type shape after torch.split: {signal_arr[0].type}")
        
    # The array of spectrograms to be returned
    ret_mel_spect_array = []
    
    # Torchaudio functional form of amplitude to db:
    # https://pytorch.org/audio/main/generated/torchaudio.functional.amplitude_to_DB.html
    
    # Attempt to replicate librosa's librosa_power_to_dB() function using API:
    # https://librosa.org/doc/main/generated/librosa.power_to_db.html 
    for elt in signal_arr:
        mel_spectrogram = torch_transformer_tutorial(elt)
        mel_spectrogram = torchaudio.functional.amplitude_to_DB(mel_spectrogram,
                                                                multiplier = 10.0,
                                                                amin = 1e-10,
                                                                db_multiplier = 1.0,
                                                                top_db=80.0) 
        ret_mel_spect_array.append(mel_spectrogram)
    
#     mel_spect_array = np.append(mel_spect_array,mel_spectrogram)
    return ret_mel_spect_array


# In[12]:


# import matplotlib
# # %matplotlib qt
# # %matplotlib inline
# matplotlib.use('Qt5Agg')

# import matplotlib.pyplot as plt
# from matplotlib import cm # import plt colormap module
# from PIL import Image # PIL required for RGBA image conversion

# import time
def get_prediction_live(input_array, base_model, seconds_recorded):
    
    # Use plt to get a decent first run exactly the same as training
    # https://stackoverflow.com/questions/73795161/can-the-output-of-plt-imshow-be-converted-to-a-numpy-array
    
    
    predictions = [] # The output array of prediction data
    time_array = [] # The output array of time it cost to run predictions
    counter = 0 # The number of predictions total performed by this function
    input_array_pre = [] # The array of preconverted input data

    classID_dict = {0 : 'air_conditioner',
                1  : 'car_horn',
                2  : 'children_playing',
                3  : 'dog_bark',
                4  : 'drilling',
                5  : 'engine_idling',
                6  : 'gun_shot',
                7  : 'jackhammer',
                8  : 'siren',
                9  : 'street_music'}

    # Time Process     
    t_start = time.time()
    
    ### - Conversion Part -
    # Conversions required to process torch tensor into RGB Image object that can be used by keras
    # convert all torch tensors to numpy ndarray
    for elt in input_array:
        input_array_pre.append(elt.numpy())


    
    # Normalize values in numpy ndarray
    # Shockingly, numpy library allows you to call functions on the entire image array
    for elt_i in range(len(input_array_pre)):
        # Numpy normalize algorithm
        # https://www.statology.org/numpy-normalize-between-0-and-1/
        input_array_pre[elt_i] = (input_array_pre[elt_i] - np.min(input_array_pre[elt_i]))/(np.max(input_array_pre[elt_i]) - np.min(input_array_pre[elt_i]))

    # Apply magma cmap
    # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    input_array_pre = cm.magma(input_array_pre)*255

    
    # Warning, image still RGBA at this point.

    # Convert all ndarrays into PIL images to pass to keras
    
    input_images = []
    
    # Elt_i is the index of cmapped RGBA images in output_cmap
    for elt in input_array_pre:
        input_images.append(Image.fromarray(np.uint8(elt)))
        
    # Convert RGBA image to RGB
    for elt_i in range(len(input_images)):
        
        # Also flip the image
        # https://pythonexamples.org/python-pillow-flip-image-vertical-horizontal/
        input_images[elt_i] = input_images[elt_i].convert('RGB').transpose(Image.Transpose.FLIP_TOP_BOTTOM)    
    
    # Clear, image no longer RGBA
    
    ### - End Conversion Part -
    
    # Entering loop
    
    # The list of reported outputs
    ret_report_list = []
    for elt in input_images:
        
        # Use keras load procedure without the image component
        test_img_array = tf.keras.utils.img_to_array(img = elt)
        
        # Reshape image array to dims = (batch_index, batch_data)   /// Warning, retrospective comment for intention.  Please check validity.
        test_img_array = tf.expand_dims(input = test_img_array, axis = 0)
        
        # Use keras NN resize layer turn 128x63 image into 128x63_scaled = 128x128
        # Asset generation occurring during runtime loop, can create this layer outside of
        # live loop for greater efficiency
        resize_layer = tf.keras.layers.Resizing(height = 128, width = 128)
        
        test_img_array = resize_layer(test_img_array)
        
        # Make predictions
        # Shape of predictions array is:
        # predictions[number of images][number of batches = 1][a vector of 10 numpy.float32 predictions]
        # prediction.append(base_model.predict(elt))
        predictions.append(base_model.predict(test_img_array))
        
        # Append time from start of process to prediction 
        time_array.append((time.time()-t_start))
        
        temp_report_string = "\n"
        
        # Walk through the predictions for this chunk and append information to report string. 
        for elt_i in range(len(predictions[counter][0])):
#             print(f"Success, len(predictions[counter][0]) is {len(predictions[counter][0])}")
            # Right justify and pad class label to length of 20 chars
            temp_report_string = temp_report_string + f"Class: {elt_i} {str(classID_dict[elt_i]) : <20} - {predictions[counter][0][elt_i] : >.5f}\n"
        
        prediction_best = np.argmax(predictions[counter])
        temp_report_string = temp_report_string + f"\nBest prediction is: {prediction_best} {classID_dict[prediction_best]}"
        temp_report_string = temp_report_string + f"\npr({classID_dict[prediction_best]}) = {np.amax(predictions[counter]) :.5f}\n"
        temp_report_string = temp_report_string + f"\npr(siren) = {predictions[counter][0][8] :.5f}\n______"

#         print(len(temp_report_string.split('\n')))
#         print(f"\nChunk: {seconds_recorded}\n {temp_report_string}") 

        # Append report for one particular chunk to the report list to be returned
        ret_report_list.append(temp_report_string)
#         print(f"Report list appended, len(temp_report_list) = {len(temp_report_list)}")

        counter += 1
    print(f"\nChunk: {seconds_recorded}\n {ret_report_list[0]}")
    # return an array of predicted probabilities
    # return an array of time cost for each prediction
    # return an array of noteworthy information from the function run
    # return an array of the spectrograms used, one per chunk, to generate predictions
    return predictions, time_array, ret_report_list, input_images


# In[ ]:





# In[13]:


def run_audio_processor_continuous(seconds_to_record, base_model, show_histogram = False):
    # The list of classes
    classID_dict = {0 : 'air_conditioner',
            1  : 'car_horn',
            2  : 'children_playing',
            3  : 'dog_bark',
            4  : 'drilling',
            5  : 'engine_idling',
            6  : 'gun_shot',
            7  : 'jackhammer',
            8  : 'siren',
            9  : 'street_music'}

    # Moved here from record_audio()
    #Reference List of Parameters
    # Load parameters
    SRATE = 16000
    # T_INTERVAL = 1/SRATE
    CHUNK = 1024 # FRAMES_PER_BUFFER = CHUNK = normally 1024 
    FORMAT = pyaudio.paInt16
    NUM_CHANNELS = 1
    # STRIDE = 2048
    # VOLUME = .25 

    # Instantiate PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=NUM_CHANNELS,
                    rate=SRATE,
                    input=True,
                    output=True,
                    )

    # num_chunks_to_record = seconds_to_record*SRATE/CHUNK
    # Modified for continuous stream
    num_chunks_to_record = 1*SRATE/CHUNK
    
    seconds_recorded = 0

    # Create torch transformer for mel spectrograms
    # Parameters
    torch_frame_size = 512
    torch_hop_length = 256
    torch_sr = 16000

    # Create torch transformer
    torch_transformer_tutorial = T.MelSpectrogram(
    sample_rate=torch_sr,
    n_fft=torch_frame_size,
    win_length=torch_frame_size,
    hop_length=torch_hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=128,
    mel_scale="htk")
    
    # The last 5 seconds array of 1 second frames 
    last_5_frames = []
    
    # Create and add 5 empty images using pillow
    # https://stackoverflow.com/questions/12760389/how-can-i-create-an-empty-nm-png-file-in-python
    last_5_frames.append(Image.new('RGB', (128, 63)))
    last_5_frames.append(Image.new('RGB', (128, 63)))
    last_5_frames.append(Image.new('RGB', (128, 63)))
    last_5_frames.append(Image.new('RGB', (128, 63)))
    last_5_frames.append(Image.new('RGB', (128, 63)))
    
    # The mapper for the ring buffer
    ring_buffer_map = [0, 1, 2, 3, 4]
    ring_i_zero = 0
    
    # Instantiate figure
    fig_spect, axs = plt.subplots(nrows = 3,
                            ncols = 5,
                            #width_ratios=[1, 1],
                            #squeeze=True,
                            constrained_layout=False,
                            figsize=(2, 6))
    
    # Set supertitle for figure
    fig_spect.suptitle("Model Spectrogram Input")
    
    # Set spacing between top and bottom row to .2
    fig_spect.subplots_adjust(hspace = .2)
    
    # Create a gridspec for the last row histogram
    # # https://stackoverflow.com/questions/48584730/make-single-plot-from-multi-columns-in-matplotlib-subplots
    gs = GridSpec(nrows = 3, ncols = 5, figure = fig_spect)
    
    # Remove axes in the last row 
    # notations is axs[row, column]
    for i in range(5):
        axs[2,i].remove()    
        
    
    # Instantiate Histogram if histogram flag is True
    # Intial values are all zeros
    
    if show_histogram == True:
        histogram_axs = fig_spect.add_subplot(gs[2,:])
        histogram_axs.set_ylim([0,1.0])

        histogram_liveplot = histogram_axs.bar(classID_dict.values(), np.zeros((10), dtype=np.float32))
    
    
    # The frame of the spectrogram viewer that is being recorded
    frame_num = 0
    
    # The array of images to be plotted on the horizonal subplot
    # Representing a series of 1 second spectrograms across time
    # Uses PLT structure where calls to axis.imshow() returns an PLT.image object bound
    # to the subplot axis it was created by.
    viewer_img_array = []
    
    # The array of text to be set for the viewer
    viewer_text_array = []
    
    # Save predictions over time
    prediction_array = []
    
    # The history of reports generated
    report_history = []
    
    # Instantiate subplot with black images
    for count_i in range(5):

        # The offset to map to the ring space
        offset_i = (count_i + ring_i_zero)%5
        
#         print(f"Setting subplot {count_i}")
#             print(axs[0][count_i])
        
        # vmin=0, vmax=255 must be explicitly set if we are instantiating the subplot
        # with blank images because min and max are inferred automatically from the image
        viewer_img_array.append(axs[0,count_i].imshow(last_5_frames[offset_i], vmin=0, vmax=255))
        axs[0,count_i].axis('off')

        axs[1,count_i].axis('off')
        
        viewer_text_array.append(axs[1,count_i].text(x=0,
                                                     y=0,
                                                     s = f"Placeholder",
                                                     fontsize = 'xx-small'))
        prediction_array.append(0)
        
        report_history.append("")
    
    # Create Large default popup window size:
    # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    
    # Change default start location for popup window
    # https://stackoverflow.com/questions/42394076/how-can-i-change-the-default-window-position-of-a-matplotlib-figure
    
    figManager = plt.get_current_fig_manager()
    figManager.window.setGeometry(50,50, 1600, 800) # (x_loc, y_loc, width, height) In pixels
    
#     # Loading complete starting prediction
#     print("Loading complete starting prediction_1")    

    # Show PLT
    plt.show(block=False)
    
    # PLT Interactive Mode On
    plt.ion()
    # plt.show(block=False)
    
    # The start time of the full processor loop
    full_run_t_start = time.time()
    
    # Loading complete starting prediction
#     print("Loading complete starting prediction_2")
    
    while (seconds_recorded < seconds_to_record):  
#         print(f"Calling record_audio with: num_chunks_to_record = {num_chunks_to_record}" )
#         print(f"Calling record_audio with: CHUNK = {1024}" )

        # - Processing Section -
    
        output = record_audio(seconds_to_record = 1,
                              stream = stream,
                              num_chunks_to_record = num_chunks_to_record,
                              CHUNK = 1024)
        output_images = create_mel_spectrogram(output, 16000, torch_transformer_tutorial)
        predictions_output, times, reports, spectrograms_used = get_prediction_live(output_images, base_model, seconds_recorded)
        
        # - End Processing Section -
        
        # Print report of classification
#         print(f"Second {seconds_recorded}\n" + reports[0])

        # Determine frame number
        # Avoid modulus with maybe faster implementation?

        # viewer_window_number = 5
        while (frame_num >= 5):
            frame_num = frame_num - 5
        
        # Update the 
        # print(f"frame_num is: {frame_num}")
        last_5_frames[frame_num] = spectrograms_used[0]
        
        prediction_array[frame_num] = predictions_output[0]
        
        report_history[frame_num] = reports[0]
        
        
        # The ring buffer offset reference is frame_num
        # ring_index_start = 0 + frame_num
        # ring_i_zero = 0 + frame_num
        
        
        # print(f"frame_num += 1 is: {frame_num}")
        #axs = axs.ravel()
        
        for count_i in range(5):
            
            # The offset to map to the ring space
            # Maps starting from the far right (index 4)
            # Starting from the start of the ring = the most recent index in the frame list denoted by ring_index_start
            # And walks backwards through the buffer (- count_i) , wrapping around
            # 5 is necessary to avoid negative index
            
            offset_i = (5 + frame_num - count_i)%5
            
            #print(f"Update loop count_i: {count_i} . Setting subplot offset_i: {offset_i}")
            
            # Display series progression is a countdown from the rightmost index 4 to the leftmost index 0
            # viewer_array_index = 4-count_i
            
            # Update viewer chunk image
            viewer_img_array[4-count_i].set_data(last_5_frames[offset_i])
            
            #viewer_prediction_array[count_i]
#             temp_prediction = np.argmax(viewer_prediction_array[offset_i])
            
            # Update viewer chunk title
            temp_title = f"{seconds_recorded - count_i} - Prediction was: {np.argmax(prediction_array[offset_i])}"
            axs[0, 4-count_i].set_title(temp_title)
            axs[0, 4-count_i].axis('off')
            
#             temp_str = f"Chunk {seconds_recorded - count_i}\n" + viewer_text_array[offset_i]
            # Update the text under the corresponding spectrogram
            viewer_text_array[4-count_i].set_text(f"Chunk {seconds_recorded - count_i}\n {report_history[offset_i]}")

        # If using histogram:
        if show_histogram == True:
            # Update the histogram
            # range(10) = range(len(predictions_output))
            for elt_i in range(10):
                histogram_liveplot[elt_i].set_height(predictions_output[0][0][elt_i])
            
                  # Test code for histogram input
#                 print(predictions_output[0][0][elt_i])
#                 print(type((predictions_output[0][0][elt_i])))
            
        seconds_recorded += 1
    
    
        # drawing updated values
        fig_spect.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig_spect.canvas.flush_events()

        
        
        # End of 1 second processing loop,
        # Increment the frame count
        frame_num += 1
        
        print(f"Time for loop is : {time.time() - full_run_t_start}\n_____")
        full_run_t_start = time.time()
        
        time.sleep(.05) # Pause pyplot for t = .05 seconds
    
    # Close audio stream
    stream.close()
    
    # Reprint primary command
    print(f"Run Command:\nrun_audio_processor_continuous(seconds_to_record = 5, base_model = base_model, show_histogram = False)")
    
    # Keep window open, block until user closes window
    plt.show()


# In[14]:


# import keras
# import os

print("All Functions Loaded, Loading Model.")
# # The directory where the model is saved
# model_save_path = r"N:\# GMU 2022 ML Model\UrbanSound8K\audio\trainset_second"

# # The name of the model
# base_model_name = r"base_model_3_8_2023_sigmoid-try11-2"

# # The name of the model
# base_model_name = r"base_model_3_8_2023_sigmoid-try11-2"

# Load the model
base_model = keras.models.load_model(os.path.join(model_save_path, base_model_name))
print(f"Loaded base_model name: {base_model_name}.")
print("Setup Complete.")


# In[16]:


print(f"Run Command:\nrun_audio_processor_continuous(seconds_to_record = 5, base_model = base_model, show_histogram = False)")


# In[19]:


run_audio_processor_continuous(seconds_to_record = 30, base_model = base_model, show_histogram = True)


# In[ ]:





# In[ ]:


# https://github.com/librosa/librosa/issues/478


# In[ ]:





# In[ ]:


# Plt plot live
# https://pythonprogramming.net/live-graphs-matplotlib-tutorial/

# update frame in matplotlib with live camera preview
# https://stackoverflow.com/questions/44598124/update-frame-in-matplotlib-with-live-camera-preview

# ***Update plt images over time
# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively/62880185#62880185

# PLT Set data in a loop
# https://stackoverflow.com/questions/56514228/update-pyplot-subplots-with-images-in-a-loop

