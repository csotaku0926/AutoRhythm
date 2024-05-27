# AutoRhythm

a simple web-based rhythm game with auto beat map generation features.

`processing/audio_signal.py` can generate beat maps automatically, given the music specified by players.

![demo image](https://github.com/csotaku0926/AutoRhythm/blob/main/img/demo.png)

## How to run
1. run `pip install -r requirements.txt` to make sure you have necessary libraries
2. run `cd processing && python audio_signal.py` to generate beat map for the music
    - I will add an interface to select song in future 
3. run `cd .. && python server.py <port>` 
4. connect to `127.0.0.1:<port>` (`port` default is 8080)

## TODO
- **Add song selecting interface**
- Remove dependency of `sklearn.cluster` (for KMeans)

## Details
This repos is an unofficial implementation of IEEE paper --

"[AutoRhythm: A music game with automatic hit-time generation and percussion identification](https://ieeexplore.ieee.org/document/7177487)"

For more technical details, please feel free to check out my [HackMD writeup](https://hackmd.io/@csotaku0926/AutoRhythm)