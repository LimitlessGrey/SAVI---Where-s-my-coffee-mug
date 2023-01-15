#!/usr/bin/env python3

import os
from gtts import gTTS
from gtts import lang
import vlc

def TTS(length, width, depth):

    tts = gTTS(f'The object is {length} centimeters tall, {width} centimeters wide and {depth} centimeters deep.',lang='en', tld='co.in')
    tts.save('path/player.mp3')

    p = vlc.MediaPlayer("path/player.mp3")

    return p.play()


