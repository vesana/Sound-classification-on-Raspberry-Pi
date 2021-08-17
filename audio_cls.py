import sounddevice as sd
import tensorflow as tf
import soundfile as sf
import numpy as np
import telegram
import pyaudio
import resampy
#import params
import wave
import time
import sys
import csv
import io

# Set up the telegram bot.
bot = telegram.Bot(token = "") # <- Bot token here

# Set the model path.
interpreter = tf.lite.Interpreter("yamnet_lite.tflite") # /home/pi/Documents/tf_lite/

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]["index"]
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]["index"]
embeddings_output_index = output_details[1]["index"]
spectrogram_output_index = output_details[2]["index"] 

file_name = "sample.wav" 
fs = 44100
duration = 2
    
def recSD():
    sd.default.device = 2
    sd.default.channels = 1, 2
    print("Recording...")
    aanifile = sd.rec(int(duration * fs), samplerate=fs, dtype="int16")
    sd.wait()
    print("Done.")
    sf.write(file_name, aanifile, fs)

def recPyAudio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    p = pyaudio.PyAudio()

    print("Recording...")
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=fs,
                input=True,
                #input_device_index=2,
                frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Done.")

def main():
    sndDevice = int(input("(1) PyAudio\n(2) SoundDevice\n>>>"))
    while True:
        try:
            if sndDevice == 1:
                recPyAudio()
            elif sndDevice == 2:
                recSD()
                
            # Decode the WAV file.
            #wav_data, sr = sf.read(file_name, dtype=np.int16)
            wf = wave.open(file_name, "rb")
            sr = wf.getframerate()
            wf.close()
            wav_data = np.fromfile(file_name, dtype=np.int16)
            assert wav_data.dtype == np.int16, "Bad sample type: %r" % wav_data.dtype
            waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
            waveform = waveform.astype("float32")

            # Convert to mono and the sample rate expected by YAMNet.
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            if sr != 16000: #params.sample_rate
                waveform = resampy.resample(waveform, sr, 16000) #params.sample_rate
               
            interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform)
            interpreter.invoke()
            scores, embeddings, spectrogram = (
                    interpreter.get_tensor(scores_output_index),
                    interpreter.get_tensor(embeddings_output_index),
                    interpreter.get_tensor(spectrogram_output_index))
            
            # Get class names.
            class_names = class_names_from_csv(open("yamnet_class_map.csv").read()) # /home/pi/Documents/tf_lite/
            
            # Report the highest-scoring classes and their scores.
            prediction = np.mean(scores, axis=0)
            top5_i = np.argsort(prediction)[::-1][:5]        
            print(file_name, ':\n' + '\n'.join('  {:12s}: {:.3f}'.format(class_names[i], prediction[i])
            for i in top5_i))

            # Check if the trigger is in the top5 highest scoring classes and send a notification if it is.
            if 378 in top5_i:
                print("Typing detected, sending telegram...")
                bot.send_message(text = "Typing detected.", chat_id=None) # <- Your ChatID here.
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nQuit")
            sys.exit(0)

        except:
            print("Failed to record, retrying")
            time.sleep(1)
              
def class_names_from_csv(class_map_csv_text):
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]
    return class_names

if __name__ == "__main__":
    main()