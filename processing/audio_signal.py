import numpy as np
import librosa
from scipy import signal
from scipy.io.wavfile import write
# plot
import matplotlib.pyplot as plt
# just to measure time elapsed
import time

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

def plot_spectrogram(x, fs=22050, H=1024, save_title=None, log=True):
    if (log):
        x = np.log(1 + 100 * x)
    
    plt.figure()
    librosa.display.specshow(x, sr=fs, hop_length=H, x_axis='time', y_axis='linear')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrom")
    
    if (save_title):
        plt.savefig(save_title)
    
    plt.show()


def compute_spectrom(x, fs=22050, N=2048, H=1024):
    X = timer(librosa.stft, "stft", 
              y=x, n_fft=N, hop_length=H, win_length=N, window='hann',
                center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    return Y

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
                fs=22050, N=2048, H=1024):
    """
    reconstruct harmony and percussive sound, given masks
    """
    X_h = X * m_h
    X_p = X * m_p

    x_h = librosa.istft(stft_matrix=X_h, hop_length=H, win_length=N, window='hann', center=True, length=x_len)
    x_p = librosa.istft(stft_matrix=X_p, hop_length=H, win_length=N, window='hann', center=True, length=x_len)

    # write ndarray to wav file
    write("harmony.wav", fs, x_h)
    write("percussive.wav", fs, x_p)

def hpss():
    x, sample_rate = timer(librosa.load, "load", 
                                path="../audio/snow_halation.mp3", duration=32)
    x_len = x.size
    l_h, l_p = 23, 29
    N, H = 2048, 1024

    y = compute_spectrom(x, N=N, H=H)
    
    m_soft_h, m_soft_p, y_h, y_p = hp_medfilt(y, l_h, l_p)

    plot_spectrogram(y_h, H=H)
    plot_spectrogram(y_p, H=H)

    reconstruct(y, m_soft_h, m_soft_p, x_len, N=N, H=H)

if __name__ == "__main__":
    hpss()