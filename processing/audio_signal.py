import numpy as np
import librosa
from scipy import signal
from scipy.fftpack import dct
from utils import *
from sklearn.cluster import KMeans

def compute_stft(x, fs=22050, N=2048, H=1024):
    X = timer(librosa.stft, "stft", 
              y=x, n_fft=N, hop_length=H, win_length=N, window='hann',
                center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    return X, Y

def cal_mfcc(X:np.ndarray, sr=22050, N=2048, H=1024,
              num_filt=40, num_cep=12, cep_lifter=None,
              do_plot=False):
    """
    calculate Mel-Frequency Cepstral Coefficient

    number of frames is `duration * fs / H`
    - `X`: STFT result (power spectrum), shape: (1 + N / 2, duration * fs / H)
    - `num_filt`: number of filters used in filter bank
    - `num_cep`: number of MFCCs
    - `cep_lifter`: D coefficient in sinusodial liftering. Default equals to `num_cep`

    returns
    - `mfcc`: (num_frames, num_cep) shape
    """
    epsilon = np.finfo(float).eps
    # apply filter banks to STFT result
    low_freq_mel = 0
    high_freq_mel = hz_to_mel(1 + sr / 2)
    # equally spaced out in mel-scaled
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filt + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N + 1) * hz_points / sr)

    fbank = np.zeros((num_filt, int(np.floor(1 + N / 2))))
    for m in range(1, num_filt + 1):
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
    # mean norm
    filter_banks -= np.mean(filter_banks, axis=0) + epsilon

    ## need to do the transpose to `filter_banks` to fit librosa's "plot_spectrogram"
    if (do_plot):
        plot_spectrogram(filter_banks.T, save_title="spectrogram.png", H=H)

    # Now compute MFCCs from spectrogram
    # (num_frames, num_cep)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_cep + 1)] # keep 2-13 coefficnets
    
    # sinusodial filtering
    cep_lifter = num_cep if not cep_lifter else cep_lifter
    lift = 1 + cep_lifter / 2 * np.sin(np.pi * np.arange(num_cep) / cep_lifter)
    mfcc *= lift

    # mean norm
    mfcc -= np.mean(mfcc, axis=0) + epsilon

    if (do_plot):
        plot_spectrogram(mfcc.T, save_title="mfcc.png", H=H)

    return mfcc

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
    else:
        x_h = librosa.istft(stft_matrix=X_h, hop_length=H, win_length=N, window='hann', center=True, length=x_len)
        x_p = librosa.istft(stft_matrix=X_p, hop_length=H, win_length=N, window='hann', center=True, length=x_len)

        # write ndarray to wav file
        write("harmony.wav", fs, x_h)
        write("percussive.wav", fs, x_p)

        return x_h, x_p
    
def salient_freq(X:np.ndarray, sr=22050, N=2048, H=1024, 
                 start_time=None, end_time=None, my_sfreq_num=None, thres_int=16):
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
    - `thres_int`: threshold interval (in sec)
    """

    start_t_band = int(start_time * sr / H) if start_time else None   
    end_t_band = int(end_time * sr / H) if end_time else None
    X = X[:, start_t_band:end_t_band]
    thres_frames = int(np.ceil(thres_int * sr / H))

    # get sum for every `thres_frames` seconds
    # m_sums = np.sum(X, axis=-1) * -1
    m_sums = np.add.reduceat(X, 
                             np.arange(0, len(X[0]), thres_frames),
                             axis=-1) * -1
    
    # top-k magnitude
    sfreq_num = int(X.shape[0] * 0.15) if not my_sfreq_num else my_sfreq_num
    
    # salient freq for every `thres_int` seconds
    # shape : (sfreq_num, duration / thres_int)
    sfreq = np.argpartition(m_sums, kth=sfreq_num, axis=0)[:sfreq_num, :]
    # sfreq = np.argpartition(m_sums, kth=sfreq_num, axis=0)[:sfreq_num]

    # then, avg power magnitude of sfreqs are summed up
    # calculate threshold every `thres_int` seconds
    
    X_sfreq = np.zeros(sfreq.shape)
    for t in range(len(X[0])):
        sfreq_in_t = sfreq[:, t // thres_frames]
        X_sfreq[:, t // thres_frames] += X[sfreq_in_t, t]
        
    thresholds = np.zeros(len(sfreq[0]))
    for t in range(len(sfreq[0])):
        thresholds[t] += np.sum(X_sfreq[:, t])
    thresholds /= thres_frames

    # expand dimension to (sfreq_num, duration_frame)
    sfreq_per_frame = np.empty((len(sfreq), len(X[0])))
    thres_per_frame = np.empty(len(X[0]))
    indices = np.arange(len(X[0])) // thres_frames
    sfreq_per_frame = sfreq[:, indices]
    thres_per_frame = thresholds[indices]

    return sfreq_per_frame, thres_per_frame

def onset_detection(X:np.ndarray, sr=22050, N=2048, H=1024, interval=0.08, n_track=4, my_sfreq_num=None,
                    start_time=None, end_time=None, thres_int=16):
    """
    identify whether there's a percussion sound
    for 32 sec, sr=22050, H=1024, one time unit is 0.046 sec, min interval is 0.07
    that is, check percussion sound every two time units

    - `X`: stft output with shape ((1 + n_fft/2), (duration * sr / hop_length))
    - `interval`: interval of detection (in sec). Default is 0.08 sec
    - `start_time`, `end_time`: refer to `salient_frequnecy()`
    - `thres_int`: threshold interval (in sec)

    returns
    - percussion: time that percussion happen
    - track_id: corresponding track ID (according to k-means on MFCCs)
    """
    y = np.abs(X) ** 2
    # cal_mfcc(y.T, sample_rate, N, H)
    mfcc = timer(cal_mfcc, "mfcc", 
                 X=y.T, sr=sr, N=N, H=H)
    mfcc = mfcc.T # (num_cep, n_frames)
    
    sfreq, thresholds = salient_freq(y, sr, N, H, start_time, end_time, my_sfreq_num, thres_int)
    # (sfreq_num, duration_frame), (duration_frame)

    time_per_frame = H / sr
    n_frame_per_interval = int(np.ceil(interval / time_per_frame))
    n_interval = int(np.ceil(len(y[0]) / n_frame_per_interval))
    
    y_sfreq_frames = np.empty((len(sfreq), n_interval))
    thres_frames = np.empty(n_interval)

    for t in range(n_interval):
        y_sfreq_frames[:, t] = y[sfreq[:, t * n_frame_per_interval], t * n_frame_per_interval]
        thres_frames[t] = thresholds[t * n_frame_per_interval]

    # compared with previous sum
    sums = np.sum(y_sfreq_frames, axis=0)
    prev_sum = np.roll(sums, 1)
    prev_sum[0] = np.inf

    # indices where sum exceeds threshold and larger than previous magnitude
    valid_indices = np.all([(sums > thres_frames), (sums > prev_sum)], axis=0)

    interval_indices = np.arange(0, len(y[0]), n_frame_per_interval)
    valid_frames = interval_indices[valid_indices]
    percussion = valid_frames * time_per_frame

    # use k-mean on MFCC to do grouping
    valid_mfcc = mfcc[:, valid_frames]
    # make it (num_valid_frames, num_cep)
    valid_mfcc = valid_mfcc.T

    kmeans = KMeans(n_clusters=n_track, random_state=0).fit(X=valid_mfcc)

    return percussion, kmeans.labels_

def main(audio_filename="../audio/snow_halation.mp3", duration=None):
    l_h, l_p = 23, 9
    N, H = 512, 256
    emp_c = 0.97

    x, sample_rate = timer(librosa.load, "load", 
                                path=audio_filename, duration=duration)
    # pre-emphasis: y(t) = x(t) - a * x(t-1)
    x = np.append(x[0], x[1:] - emp_c * x[:-1])

    # x.shape: (duration * sr, )
    x_len = x.size
    print("duration (in sec):", x_len / sample_rate)

    X, y = compute_stft(x, N=N, H=H) 

    # HPSS
    m_soft_h, m_soft_p, _, _ = hp_medfilt(y, l_h, l_p)
    X_h, X_p = reconstruct(X, m_soft_h, m_soft_p, x_len, N=N, H=H, do_istft=False)
    
    percussion_h, track_h = timer(onset_detection, "onset detection (h)", 
                         X=X_h, N=N, H=H, n_track=4, thres_int=8)
    percussion_p, track_p = timer(onset_detection, "onset detection (p)", 
                         X=X_p, N=N, H=H, n_track=4, thres_int=8)

    # merge percussion and harmony
    all_p, all_t = merge_beatmap(percussion_h, track_h, percussion_p, track_p + 4)
    output_json(all_p, all_t, stop_time=int(np.ceil(x_len / sample_rate)) ,filename="whole.json")
    
if __name__ == "__main__":
    main(audio_filename="../audio/snow_halation.mp3", duration=None)
