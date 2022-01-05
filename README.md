# Sound classification and notification project on Raspberry Pi 4 using Respeaker 2-mic Pi Hat, Tensorflow, YAMNet and Telegram. 
Records 2 seconds of audio (by default), then analyzes the 5 most probable sound sources and returns their names and probabilities.
Sends a message on Telegram if a specific sound is detected. You can specify the trigger in line
*if 378 in top5_i and useTG == True:
378 is "Typing" in yamnet_class_map.csv, and can be changed to the desired value, for example dog barking or gun sounds etc.

Integration with Azure on the works to provide visualization of some kind (graphs, spectogram?) via a web app.

## Installation
Install packages with pip:
'''
pip install -r requirements.txt
'''

## Notes
Tested on Windows 10 with Python 3.9.9 and Raspian with Python 3.7.3. Tensorflow is not currently available for Python 3.10.
Telegram uses environment variables TG_BOT and TG_CHATID.

# YAMNet
YAMNet is a pretrained deep net that predicts 521 audio event classes based on
the [AudioSet-YouTube corpus](http://g.co/audioset), and employing the
[Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable
convolution architecture.