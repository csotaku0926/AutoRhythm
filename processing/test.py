import numpy as np
import librosa
import matplotlib.pyplot as plt

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

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def amplitude_to_db(amp):
    return 20 * np.log10(amp)

def calc(X:np.ndarray, sr=22050, N=2048, H=1024,
              frame_size=0.025):
    
    # framing: split signal into short-time frames
    # `frame_step` is known as `hop_length`
    frame_length, frame_step = int(round(frame_size * sr)), H #int(round(frame_stride * sr))
    signal_length = len(X)
    # at least have one frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad = np.zeros(num_frames * frame_step + frame_length - signal_length)
    pad_signal = np.append(X, pad)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    # frames shape: (num_frames , frame_length)
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # apply Hamming window (0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1)))
    frames *= np.hamming(frame_length)

    # STFT
    mag_frames = np.abs(np.fft.rfft(frames, N)) # magnitude of FFT
    pow_frames = (mag_frames ** 2) / N # power spectrum

    return pow_frames

def cal_mfcc(X:np.ndarray, sr=22050, N=2048, H=1024,
              nfilt=40):
    """
    calculate Mel-Frequency Cepstral Coefficient
    - `X`: STFT result (power spectrum), shape: (1 + N / 2, duration * fs / H)
    - `nfilt`: number of filters used in filter bank
    """
    epsilon = np.finfo(float).eps
    # apply filter banks to STFT result
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(1 + sr / 2)
    # equally spaced out in mel-scaled
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N + 1) * hz_points / sr)

    fbank = np.zeros((nfilt, int(np.floor(1 + N / 2))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1]) # f(m - 1), left
        f_m = int(bins[m])           # f(m), center
        f_m_plus = int(bins[m + 1])  # f(m + 1), right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    
    filter_banks = np.dot(X, fbank.T)
    # if there's 0, replace with epsilon
    filter_banks = np.where(filter_banks == 0, epsilon, filter_banks)
    # there, we get the spectrogram
    filter_banks = amplitude_to_db(filter_banks)

    plot_spectrogram(filter_banks.T, sr, H)

if __name__ == "__main__":
    N, H = 512, 256
    emp_c = 0.97
    audio_filename = "../audio/snow_halation.mp3"

    x, sample_rate = librosa.load(path=audio_filename)
                                
    # pre-emphasis: y(t) = x(t) - a * x(t-1)
    x = np.append(x[0], x[1:] - emp_c * x[:-1])
    X = calc(x, N=N, H=H)
    cal_mfcc(X, N=N, H=H)