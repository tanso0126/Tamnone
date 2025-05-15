#!/bin/bash

# 아래 코드 복사해서 셸에 붙여넣기
python3 whisper_online_server.py --language ko --min-chunk-size 1 --vad --buffer_trimming_sec 1 --model medium

# 아래 코드는 새로운 터미널 열어서 붙여넣기
sox -t coreaudio default -b 16 -e signed -c 1 -r 16000 -t raw - | nc localhost 43007