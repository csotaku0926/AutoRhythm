import numpy as np
import librosa
from scipy import signal
# plot
import matplotlib.pyplot as plt
# measure time elapsed
import time
# io
from scipy.io.wavfile import write
import os
import json

def np_stft(signal, window_size, hop_length):
    """
    Short-time Fourier Transform (STFT) with Numpy
    It does the same thing as `librosa.stft`, but slower
    """
    n_frames = 1 + (len(signal) - window_size) // hop_length
    stft_matrix = np.empty((window_size // 2 + 1, n_frames), dtype=complex)

    for i in range(n_frames):
        frame = signal[i * hop_length : i * hop_length + window_size]
        # hamming window (window function)
        window_frame = frame * np.hamming(window_size)
        stft_matrix[:, i] = np.fft.rfft(window_frame)

    return stft_matrix

def timer(callback, title, **kwargs):
    """
    execute callback function and count elapsed time
    """
    start_time = time.time()
    ret = callback(**kwargs)
    end_time = time.time()
    print(f"[*] {title} time: {end_time-start_time}")
    return ret

def plot_spectrogram(x:np.ndarray, fs=22050, H=1024, save_title=None, log=True):
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


def compute_stft(x, fs=22050, N=2048, H=1024):
    X = timer(librosa.stft, "stft", 
              y=x, n_fft=N, hop_length=H, win_length=N, window='hann',
                center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    return X, Y

def hp_medfilt(y, l_h, l_p, binary_mask=False):
    """
    Compute horizontal and vertical median filtering
    using `scipy.signal.medfilt`
    then, use Wiener masking to assign each frequency bin to either m_h or m_p
    
    - `y`: power spectrogram 
    - `l_h`: filter length for horizontal filtering
    - `l_p`: filter length for vertical filtering
    - `binary_mask`: binary mask or soft mask
    returns
    - `m_h`: mask of harmonic sound
    - `m_p`: mask of percussive sound
    """
    y_h = signal.medfilt(y, [1, l_h])
    y_p = signal.medfilt(y, [l_p, 1])
    
    m_h, m_p = None, None
    if (binary_mask):
        # binary masking
        m_h = np.int8(y_h >= y_p)
        m_p = np.int8(y_h < y_p)
        
    else:
        # soft masking
        eps = 1e-4
        m_h = (y_h + eps / 2) / (y_h + y_p + eps)
        m_p = (y_p + eps / 2) / (y_h + y_p + eps)
    
    return m_h, m_p, y_h, y_p

def reconstruct(X, m_h, m_p, x_len,
                fs=22050, N=2048, H=1024,
                do_istft=False):
    """
    reconstruct harmony and percussive sound, given masks

    - `do_istft`: perform inverse stft or not. This should only be used to listen to output (get .wav)
    """
    X_h = X * m_h
    X_p = X * m_p

    if (not do_istft):
        return X_h, X_p
    
    x_h = librosa.istft(stft_matrix=X_h, hop_length=H, win_length=N, window='hann', center=True, length=x_len)
    x_p = librosa.istft(stft_matrix=X_p, hop_length=H, win_length=N, window='hann', center=True, length=x_len)

    # write ndarray to wav file
    write("harmony.wav", fs, x_h)
    write("percussive.wav", fs, x_p)

    return x_h, x_p
    
def salient_freq(X:np.ndarray, sr=22050, H=1024, start_time=None, end_time=None, my_sfreq_num=None):
    """
    capture frequencies with top 15% magnitude

    np.abs(X) is the magnitude of frequency bin `f` at frame `t`
    bin `f` correspoonds to frequecies (0, sr/n_fft, 2*sr/n_fft, ..., sr/2) 
    time `t` corresponds to frames[i] * hop_length

    - `X`: stft output with shape ((1 + n_fft/2), (duration * sr / hop_length))
    
    Note: In paper, `X` is given by user to capture sfreq for certain percussion sound
    , I will use whole audio signal for now
    - `start_time`: start time of salient frequencies (in sec)
    - `end_time`: end time of salient frequencies (in sec)
    """

    start_t_band = int(start_time * sr / H) if start_time else None   
    end_t_band = int(end_time * sr / H) if end_time else None
    X = X[:, start_t_band:end_t_band]

    x_test = librosa.istft(stft_matrix=X, hop_length=H, win_length=2048, window='hann', center=True)
    write("test.wav", sr, x_test)

    s_db = librosa.amplitude_to_db(np.abs(X), ref=np.max) 
    sfreq_num = int(s_db.shape[0] * 0.15) if not my_sfreq_num else my_sfreq_num
    m_sum = np.sum(s_db, axis=-1) * -1
    # top-k magnitude
    sfreq = np.argpartition(m_sum, kth=sfreq_num)[:sfreq_num]

    # then, avg power magnitude of sfreqs are summed up and mutiplied by 0.4
    threshold = np.sum(np.mean(X[sfreq], axis=1)) * 0.4
    
    return sfreq, threshold

def onset_detection(X:np.ndarray, sr=22050, H=1024, interval=0.08,
                    start_time=None, end_time=None):
    """
    identify whether there's a percussion sound
    for 32 sec, sr=22050, H=1024, one time unit is 0.046 sec, min interval is 0.07
    that is, check percussion sound every two time units

    - `X`: stft output with shape ((1 + n_fft/2), (duration * sr / hop_length))
    - `interval`: interval of detection (in sec). Default is 0.08 sec
    - `start_time`, `end_time`: refer to `salient_frequnecy()`
    """
    sfreq, threshold = salient_freq(X, sr, H, start_time, end_time)
    # to show salient freq spectrogram 
    # X[~np.isin(np.arange(len(X)), sfreq)] = np.zeros(X.shape[-1])
    # plot_spectrogram(np.abs(X), save_title="sfreq.png")

    # then, avg power magnitude of sfreqs are summed up and mutiplied by 0.4
    # threshold = np.sum(np.mean(X[sfreq], axis=1))
    time_per_frame = H / sr
    n_frame_per_interval = int(np.ceil(interval / time_per_frame))

    sums = np.sum(X[sfreq, ::n_frame_per_interval], axis=0)

    # compared with previous
    prev_sum = np.roll(sums, 1)
    prev_sum[0] = np.inf

    # indices where sum exceeds threshold and larger than previous magnitude
    valid_indices = (sums > threshold) #np.all([(sums > threshold), (sums > prev_sum)], axis=0)

    interval_indices = np.arange(0, X.shape[1], n_frame_per_interval)
    percussion = interval_indices[valid_indices] * time_per_frame

    return percussion

def X_to_x(X, H=1024, N=2048, sample_rate=22050):
    x_test_rm = librosa.istft(stft_matrix=X, hop_length=H, win_length=N, window='hann', center=True)
    write("test.wav", sample_rate, x_test_rm)

def hpss(audio_filename="../audio/snow_halation.mp3"):
    l_h, l_p = 23, 9
    N, H = 2048, 1024

    x, sample_rate = timer(librosa.load, "load", 
                                path=audio_filename)
    # x.shape: (duration * sr, )
    x_len = x.size
    print("duration (in sec):", x_len / sample_rate)

    X, y = compute_stft(x, N=N, H=H) 

    m_soft_h, m_soft_p, _, _ = hp_medfilt(y, l_h, l_p)
    X_h, X_p = reconstruct(X, m_soft_h, m_soft_p, x_len, N=N, H=H, do_istft=False)

    percussion_h = timer(onset_detection, "onset detection (h)", X=X_h, start_time=25, end_time=30)
    percussion_p = timer(onset_detection, "onset detection (p)", X=X_p, start_time=25, end_time=30)

    return percussion_h, percussion_p
    
def output_json(percussion:np.ndarray, interval=0.05, dirname="../beat_map", filename="percussion.json"):
    """
    Write percussion array to json file used by `script/game.js`

    - `interval`: update frequency of the game, default is 50 ms
    """
    notes = [{"track": "a", "second": int(s / interval)} for s in percussion]
    output = {
        "stop_time": 2400,
        "notes": notes,
    }
    
    if (not os.path.exists(dirname)):
        os.makedirs(dirname)
    
    with open(os.path.join(dirname, filename), "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    per_h, per_p = hpss()
    output_json(per_h, filename="harmony.json")
    output_json(per_p)