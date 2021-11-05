## Sound classification and notification project on Raspberry Pi 4 using Respeaker 2-mic Pi Hat, Tensorflow, YAMNet and Telegram. 

Integration with Azure on the works to provide visualization of some kind (graphs, spectogram?) via a web app.

These need to be changed to your microphone device: 
* Line 33:  sd.default.device = 2
* Line 52:  input_device_index=2,

## How to check device indexes in Python:
For sounddevice:
```
import sounddevice as sd
sd.query_devices()
```
For pyaudio:
```
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
  print(p.get_device_info_by_index(i))
```

* Line 118: bot.send_message(text = "Typing detected.", chat_id=None) # <- Your ChatID here.
```
pip3 install sounddevice soundfile numpy resampy wave python-telegram-bot
```
# YAMNet

YAMNet is a pretrained deep net that predicts 521 audio event classes based on
the [AudioSet-YouTube corpus](http://g.co/audioset), and employing the
[Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable
convolution architecture.
