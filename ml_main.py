from math import log, sqrt
import warnings
import scipy
import tensorflow as tf
import numpy as np
import IPython.display
import librosa.display
from tqdm import tqdm
# from vscode_audio import Audio
import sounddevice as sd
import argparse
import soundfile
import os
from glob import glob

# from tensorflow.keras.layers import Conv1D,Conv1DTranspose,Concatenate,Input
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import Constraint

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from fft_basic import fft_init, fft_analysis, fft_synthesis
from fft_noise_gate import fft_noise_gate

# 8ms data frame
hop_size_dict = {
    16000:128,
    48000:384,
}

fft_size_dict = {
    16000:384,
    48000:1152,
}

def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

class WeightClip(Constraint):
    def __init__(self, c=2):
        self.c = c
    def __call__(self, p):
        return K.clip(p, -self.c, self.c)
    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

def get_features(in_fft, fragment_size, fft_bins_num):
    # in_fft = in_fft.reshape(-1)
    features = np.empty((fragment_size, fft_bins_num)) 
    for t in range(fragment_size):
        for i in range(fft_bins_num):
            amp = sqrt(in_fft[t][i].real * in_fft[t][i].real + in_fft[t][i].imag * in_fft[t][i].imag)
            features[t][i] = 20 * log(amp)
    return features

def add_file(mic_file, sample_rate, hop_size, fft_size, fragment_size):
    mic, mic_fs = soundfile.read(mic_file)
    voip_fs = mic_fs
    if len(mic.shape)>1:
        # mic = mic[:,0]
        # warnings.warn('Mono input files are suppoprted')
        return None, False
    if mic_fs != sample_rate:
        # warnings.warn('Sample rate ' + str(sample_rate) + ' only wav files are supported')
        return None, False

    wola_filter = fft_init(hop_size, fft_size)
    wola_ana_buff = np.zeros(fft_size)
    chunks = np.int32(np.ceil(len(mic) / np.float32(hop_size)))

    # Check the end of the file
    if chunks * hop_size < len(mic): 
        chunks += 1
        mic = np.concatenate((mic, np.zeros(chunks * hop_size - len(mic)))) 

    voip = np.empty(chunks * hop_size, dtype=np.float32) 
    fft_chank = np.empty(fft_size, dtype=np.float32) 
    time_chank = np.empty(hop_size, dtype=np.float32) 
    mic_fft = np.empty((chunks, fft_size // 2 + 1), dtype=np.complex_) 

    for t in range(chunks - 1):
        start = t * hop_size
        end = t * hop_size + hop_size
        time_chank = mic[start:end]
        # fft_chank, mic_ana_bfr = fb_analysis(time_chank, mic_ana_bfr, ana_filter, fft_size)
        fft_chank, wola_ana_buff = fft_analysis(time_chank, wola_ana_buff, wola_filter, fft_size)
        mic_fft[t:] = fft_chank
        # time_chank, out_syn_bfr = fb_synthesis(fft_chank, out_syn_bfr, syn_filter, hop_size)
        # voip[start:end] = time_chank
    return mic_fft, True

def check_file(wav_file, sample_rate, fft_bins_num):
    wav, wav_fs = soundfile.read(wav_file)
    if len(wav.shape)>1:
        return None, False
    if wav_fs != sample_rate:
        return None, False
    length = len(wav) 
    chunks = np.int32(np.ceil(length / np.float32(fft_bins_num)))
    return chunks, True



def get_train(root_folder, sample_rate, hop_size, fft_size, batch_size, fragment_size, fft_bins_num):
    wav_dir = os.path.join(root_folder, '')
    # signal_train = [] 
    signal_train = np.empty((batch_size, int(fragment_size), int(fft_bins_num)), dtype=np.float32)
    t = 0
    for fn in tqdm(os.listdir(wav_dir), desc = "Train"):
        wav_fn = os.path.join(wav_dir, fn)
        data, status = add_file(wav_fn, sample_rate, hop_size, fft_size, fragment_size)
        if (status):
            features = get_features(data, fragment_size, fft_bins_num)
            # features = features.reshape(-1, 1)
            signal_train[t,] = features
            t += 1
    return signal_train

def scan_folder(root_folder, sample_rate, fft_bins_num):
    wav_dir = os.path.join(root_folder, '')
    num = 0
    min_chunks = 10000000
    for fn in tqdm(os.listdir(wav_dir), desc = "Scanning"):
        wav_fn = os.path.join(wav_dir, fn)
        chunks, status = check_file(wav_fn, sample_rate, fft_bins_num)
        if (status):
            num += 1
            if (min_chunks > chunks):
                min_chunks = chunks
    return min_chunks, num

def get_dataset(x_train, y_train, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # dataset = dataset.shuffle(100).batch(batch_size, drop_remainder=True)
    return dataset

def GetMopdel(batch, bins):
    constraint = WeightClip(0.499)
    reg = 0.000001
    batch = 1 
    mdl_input = Input(name='mdl_input', shape=(1, bins), batch_size=batch)
    noise_gru = GRU(bins, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(mdl_input)
    denoise_output = Dense(bins, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(noise_gru)
    model = Model(inputs=mdl_input, outputs=[denoise_output])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.002),loss=tf.keras.losses.MeanAbsoluteError(),run_eagerly=True)
    model.summary()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ocata Audio processing')
    parser.add_argument('--wav_dir', type=str, default='wav/input', help='directory containing input wavfiles')
    parser.add_argument('--ref_dir', type=str, default='wav/reference', help='directory containing reference wavfiles')
    parser.add_argument('--out_dir', type=str, default='wav/output', help='directory containing output wavfiles')
    parser.add_argument('--clean_dir', type=str, default='wav/CleanData', help='directory containing clean wavfiles')
    parser.add_argument('--noise_dir', type=str, default='wav/NoisyData', help='directory containing noisy wavfiles')
    args, _ = parser.parse_known_args()

    sample_rate = int(16000)
    buffer_size = int(8) 
    hop_size = int(sample_rate * buffer_size / 1000 )
    fft_size = int(hop_size * 3) 
    fft_bins_num = fft_size // 2 + 1
    band_num = fft_bins_num

    fragment_size, batch_size = scan_folder(args.clean_dir, sample_rate, fft_bins_num)
    clean = get_train(args.clean_dir, sample_rate, hop_size, fft_size, batch_size, fragment_size, fft_bins_num)
    noisy = get_train(args.noise_dir, sample_rate, hop_size, fft_size, batch_size, fragment_size, fft_bins_num)

    batch = 1
    model = GetMopdel(batch=batch, bins=fft_bins_num)

    clean = clean.reshape(clean.shape[0] * clean.shape[1], clean.shape[2])
    noisy = noisy.reshape(noisy.shape[0] * noisy.shape[1], noisy.shape[2])
    # train_dataset = get_dataset(clean, noisy, batch_size)
    history = model.fit(noisy, clean, batch_size=batch, epochs=2, verbose=2, shuffle=False)

