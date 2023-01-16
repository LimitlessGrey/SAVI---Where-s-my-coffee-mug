#!/usr/bin/env python3

import os
from gtts import gTTS
from gtts import lang
import vlc

def TTS(string):

    tts = gTTS(string,lang='en', tld='co.in')
    tts.save('assignment_2/SAVI---Where-s-my-coffee-mug/player.mp3')

    p = vlc.MediaPlayer("assignment_2/SAVI---Where-s-my-coffee-mug/player.mp3")

    return p.play()


