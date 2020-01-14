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
            
            newArray = np.append(aux, mfccs)
            newArray = np.append (newArray, librosa.feature.delta(mfccs))
            newArray = np.append(newArray, librosa.feature.delta(mfccs, order=2))
            
            aux=[]
            aux.append(newArray)
            df = df.append(aux, ignore_index=True)
    df.to_csv('prueba.csv', index = None, header=True)
    print("FINNN")

 