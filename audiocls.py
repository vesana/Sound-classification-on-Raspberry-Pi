#from azure.iot.device.aio import IoTHubDeviceClient
#from azure.iot.device import Message
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd
import tensorflow as tf
import soundfile as sf
import numpy as np
import telegram
import pyaudio
import resampy
import asyncio
import json
import wave
import time
import sys
import csv
import os
import io

# Ask if to use Telegram messaging.
askTG = str(input("Use Telegram? (y/n): "))
if askTG == "y":
    useTG = True
    try:
        bot = telegram.Bot(token = os.getenv("TG_BOT"))
    except:
        print("Couldn't find environment variable 'TG_BOT' for bot token, skipping Telegrams.")
        useTG = False
elif askTG == "n":
    useTG = False
else: 
    useTG = False

# Ask microphone index number and set some defaults.
print("\n", sd.query_devices())
micDev = int(input("\nSelect your microphone/soundcard: "))
file_name = "sample.wav" 
fs = 44100
duration = 2

# Set the model path.
interpreter = tf.lite.Interpreter("yamnet_lite.tflite")
input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]["index"]
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]["index"] 

async def main():
    # Select the audio library to use.
    sndDevice = int(input("\nSelect the audio library to use:\n(1) PyAudio\n(2) SoundDevice\n>>>"))
    while True:
        try:
            if sndDevice == 1:
                recPyAudio()
            elif sndDevice == 2:
                recSD()
            else:
                break
                
            # Decode the WAV file.
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
            if sr != 16000:
                waveform = resampy.resample(waveform, sr, 16000)
               
            interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform)
            interpreter.invoke()
            scores = interpreter.get_tensor(scores_output_index)
            
            # Get class names.
            class_names = class_names_from_csv(open("yamnet_class_map.csv").read())
            
            # Report the highest-scoring classes and their scores.
            prediction = np.mean(scores, axis=0)
            top5_i = np.argsort(prediction)[::-1][:5]        
            print(file_name, ':\n' + '\n'.join('  {:12s}: {:.3f}'.format(class_names[i], prediction[i])
            for i in top5_i))

            # Check if the trigger is in the top5 highest scoring classes and send a notification if it is. 378 is typing on a computer keyboard.
            if 378 in top5_i and useTG == True:
                print("Typing detected, sending telegram...")
                bot.send_message(text = "Typing detected.", chat_id=os.getenv("TG_CHATID"))
                time.sleep(0.5)

            classes = {
                "class": class_names[top5_i[0]],
                "probability": float("{0:.3f}".format(prediction[top5_i[0]]))}

            for i in top5_i:
                prediction[i] = round(prediction[i], 3)
                
            s = json.dumps(classes)
            #specto()
            #await sendHub(s)
            
        except KeyboardInterrupt:
            os.remove("sample.wav")
            print("\nQuit")
            sys.exit(0)

        except:
            print("Failed to record, retrying")
            time.sleep(1)

def specto():
    '''Displays a spectrogram.'''
    sample_rate, data = wavfile.read("sample.wav")
    sample_freq, segment_time, spec_data = signal.spectrogram(data, sample_rate)  
    plt.pcolormesh(segment_time, sample_freq, spec_data)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def recSD():
    sd.default.device = micDev
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
                input_device_index=micDev,
                frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(fs / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Done.")

def class_names_from_csv(class_map_csv_text):
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]
    return class_names

async def sendHub(top5):
    '''Sends the highest scoring sound name and its probability to Azure IoTHub'''
    print("Send:", top5)

    # Connection string from ENV variable
    conn_str = os.getenv("AZ_CONNSTRING")
    device_client = IoTHubDeviceClient.create_from_connection_string(conn_str)
    await device_client.connect()

    # Send message to IoTHub
    msg = Message(top5)
    msg.content_encoding = "utf-8"
    msg.content_type = "application/json"  
    print("Sending message to Hub...")
    await device_client.send_message(msg)
    print("Message successfully sent!")

    await device_client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())