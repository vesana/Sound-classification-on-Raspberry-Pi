from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message
import uuid
import sounddevice as sd
import tensorflow as tf
import soundfile as sf
import numpy as np
import telegram
import pyaudio
import resampy
#import params
import asyncio
import wave
import time
import sys
import csv
import os
import io

# Set up the telegram bot.
bot = telegram.Bot(token = "")

# Set the model path.
interpreter = tf.lite.Interpreter("yamnet_lite.tflite")

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
                input_device_index=2,
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

async def main():
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
                #bot.send_message(text = "Typing detected.", chat_id=1925524745)
                #time.sleep(1)

            classes = ""
            for i in top5_i:
                prediction[i] = round(prediction[i], 3)
                classes = classes + "'" + class_names[i] + ":" + str(prediction[i]) + "'" + " "
                #print(classes, prediction[i])
            classes = classes[:-1]
            await sendHub(classes)
            time.sleep(10)
            
        except KeyboardInterrupt:
            os.remove("sample.wav")
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

async def sendHub(top5):
    print("Send:", top5)

    # Connection string muuttujasta
    conn_str = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")
    device_client = IoTHubDeviceClient.create_from_connection_string(conn_str)
    await device_client.connect()

    # Send message to IoTHub
    msg = Message(str(top5))
    msg.message_id = uuid.uuid4()
    msg.content_encoding = "utf-8"
    msg.content_type = "application/json"  
    print("Sending message to Hub...")
    await device_client.send_message(msg)
    print("Message successfully sent!")

    await device_client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())