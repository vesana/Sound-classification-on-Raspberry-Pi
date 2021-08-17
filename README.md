Sound classification and notification project on Raspberry Pi 4 using 
Respeaker 2-mic Pi Hat, Tensorflow, YAMNet and Telegram. 

Line 118: bot.send_message(text = "Typing detected.", chat_id=None) # <- Your ChatID here.

pip3 install sounddevice soundfile numpy resampy wave csv telegram


# YAMNet

YAMNet is a pretrained deep net that predicts 521 audio event classes based on
the [AudioSet-YouTube corpus](http://g.co/audioset), and employing the
[Mobilenet_v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable
convolution architecture.
