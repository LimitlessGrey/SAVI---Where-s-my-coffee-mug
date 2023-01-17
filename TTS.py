#!/usr/bin/env python3

import os
from gtts import gTTS
from gtts import lang
import vlc

def TTS(string):

    tts = gTTS(string,lang='en', tld='co.in')
    tts.save('player.mp3')

    p = vlc.MediaPlayer("player.mp3")

    return p.play()


