import numpy as np
import librosa
# plot
import matplotlib.pyplot as plt
# measure time elapsed
import time
# io
from scipy.io.wavfile import write
import os
import json


def timer(callback, title, **kwargs):
    """
    execute callback function and count elapsed time
    """
    start_time = time.time()
    ret = callback(**kwargs)
    end_time = time.time()
    print(f"[*] {title} time: {end_time-start_time}")
    return ret

def plot_spectrogram(x:np.ndarray, fs=22050, H=1024, save_title=None, log=False):
    # change complex array to magnitude
    if (x.dtype == np.complex64):
        x = np.abs(x) ** 2
    
    if (log):
        x = np.log(1 + 100 * x)
    
    plt.figure()
    librosa.display.specshow(x, sr=fs, hop_length=H, x_axis='time', y_axis='linear')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrom")
    
    if (save_title):
        plt.savefig(save_title)
    
    plt.show()

def plot_waveform(x, fs=22050, save_title=None):
    plt.figure()
    librosa.display.waveshow(y=x, sr=fs)
    plt.title("Waveform")
    if (save_title):
        plt.savefig(save_title)

    plt.show()

def X_to_x(X, H=1024, N=2048, sample_rate=22050):
    x_test_rm = librosa.istft(stft_matrix=X, hop_length=H, win_length=N, window='hann', center=True)
    write("test.wav", sample_rate, x_test_rm)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def amplitude_to_db(amp):
    return 20 * np.log10(amp)

def id_to_track(id_):
    id_track = ["a", "s", "d", "f", "h", "j", "k", "l"]
    return id_track[id_]

def merge_beatmap(percussion_h, track_h, percussion_p, track_p):
    """
    merge percussion and track (sorted by time) into one sorted array respectively
    """
    all_percussion = np.empty(len(percussion_h) + len(percussion_p), dtype=percussion_h.dtype)
    all_track = np.empty(len(percussion_h) + len(percussion_p), dtype=track_h.dtype)
    
    i_h, i_p, k = 0, 0, 0
    while (i_h < len(percussion_h) and i_p < len(percussion_p)):
        if (percussion_h[i_h] < percussion_p[i_p]):
            all_percussion[k] = percussion_h[i_h]
            all_track[k] = track_h[i_h]
            i_h += 1

        else:
            all_percussion[k] = percussion_p[i_p]
            all_track[k] = track_p[i_p]
            i_p += 1
        k += 1
    
    while (i_h < len(percussion_h)):
        all_percussion[k] = percussion_h[i_h]
        all_track[k] = track_h[i_h]
        k += 1
        i_h += 1

    while (i_p < len(percussion_p)):
        all_percussion[k] = percussion_p[i_p]
        all_track[k] = track_p[i_p]
        k += 1
        i_p += 1

    return all_percussion, all_track

def output_json(percussion:np.ndarray, track_id:np.ndarray, stop_time:int, interval=0.05, 
                dirname="../beat_map", filename="percussion.json"):
    """
    Write percussion array to json file used by `script/game.js`
    """
    notes = [{"track": id_to_track(t), "second": int(s / interval)} for (t, s) in zip(track_id, percussion)]
    output = {
        "stop_time": int(stop_time / interval) + 200,
        "notes": notes,
    }
    
    if (not os.path.exists(dirname)):
        os.makedirs(dirname)
    
    with open(os.path.join(dirname, filename), "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4)