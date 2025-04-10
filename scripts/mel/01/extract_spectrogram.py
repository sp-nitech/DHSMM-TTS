import argparse
import librosa
import numpy as np
import os
import random
import sys
import wave

from config import sr, fs, wl, ft, feature_types, mfmin, mfmax, num_mels

trimming_threshold = 0.005
min_level_db = -120
requires_linear = 'linear' in feature_types
requires_mel = 'mel' in feature_types


def trim_silence(x):
    rms = np.squeeze(librosa.feature.rms(y=x, frame_length=wl, hop_length=fs))
    active_frame_indices = np.nonzero(trimming_threshold <= rms)[0]
    sample_indices = librosa.core.frames_to_samples(active_frame_indices, hop_length=fs)
    length = int(0.2 * sr)
    return x[max(0, sample_indices[0] - length) : min(len(x) - 1, sample_indices[-1] + length)]

def preprocess(x):
    x = librosa.util.buf_to_float(x, n_bytes=2, dtype=np.float32)
    x = trim_silence(x)
    return x
    
def extract_spectrogram(x):
    def _stft(y):
        return librosa.stft(y=y, n_fft=ft, hop_length=fs, win_length=wl)
    
    def _amp_to_db(S):
        min_level = np.exp(min_level_db / 20.0 * np.log(10))
        return 20.0 * np.log10(np.maximum(min_level, S))
    
    def _linear_to_mel(S):
        M = librosa.filters.mel(sr=sr, n_fft=ft, n_mels=num_mels, fmin=mfmin, fmax=mfmax)
        return np.dot(M, S)

    out = {}
    spectrogram = np.abs(_stft(x))
    if requires_linear:
        linear_spectrogram = _amp_to_db(spectrogram).astype(np.float32)
        out['linear'] = linear_spectrogram.T
    if requires_mel:
        mel_spectrogram = _amp_to_db(_linear_to_mel(spectrogram)).astype(np.float32)
        out['mel'] = mel_spectrogram.T
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scpfile', help='list file containing basenames of data')
    parser.add_argument('--wavdir', default='wav')
    parser.add_argument('--outdir', default='data/wav')
    args = parser.parse_args()
    
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    
    with open(args.scpfile, 'r') as f:
        bases = [os.path.join(*line.strip().split(' ')) for line in f.readlines()]
        
    for base in bases:
        wavfile = os.path.join(args.wavdir, base + '.wav')
        print('processing {}'.format(base))
        
        with wave.open(wavfile, 'rb') as f:
            assert sr == f.getframerate()
            assert 2 == f.getsampwidth()
            assert 1 == f.getnchannels()
            x = np.frombuffer(f.readframes(-1), dtype=np.int16)

        y = preprocess(x)
        if len(y) < len(x) // 4:
            print('Warning: waveform samples are too short after pre-processing. [{}]'.format(wavfile), file=sys.stderr)
        spectrograms = extract_spectrogram(y)
        
        wavfile = os.path.join(args.outdir, base + '.wav')
        specfile = os.path.join(args.outdir, base + '.npz')
        os.makedirs(os.path.dirname(specfile), exist_ok=True)
        
        with wave.open(wavfile, 'wb') as f:
            y = (y * 2**15).astype(np.int16)
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sr)
            f.writeframes(y)
        np.savez(specfile, **spectrograms)
        
    print('\ndone')
    
