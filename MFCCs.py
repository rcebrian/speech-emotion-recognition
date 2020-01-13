import os
import pandas as pd
import librosa
import numpy as np
import soundfile


def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = librosa.load(file_name, mono=True)
        mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
    return mfccs_mean


if __name__ == '__main__':
    df = pd.DataFrame()

    for dirs in os.listdir('./data'):
        for fi in os.listdir('./data/' + dirs + '/'):
            aux = []
            aux.append(fi.split('_', 1)[0])
            mfccs = extract_feature('./data/' + dirs + '/' + fi)
            aux.extend(mfccs)
            aux.extend(librosa.feature.delta(mfccs))
            aux.extend(librosa.feature.delta(mfccs, order=2))
            # print(aux)
            # print('>>>>>>>')
            df.append(aux)
    df.to_csv('prueba.csv')