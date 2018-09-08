import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
from scipy import signal
from pysndfx import AudioEffectsChain
import librosa

warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):

    features = mfcc.mfcc(audio,sr)

    #############################
    #                           #
    #      Noise Removal        #
    #                           #
    #############################

    features = mfcc.logfbank(audio)
    features = mfcc.lifter(features)

    sum_of_squares = []
    index = -1
    for r in features:
        sum_of_squares.append(0)
        index = index + 1
        for n in r:
            sum_of_squares[index] = sum_of_squares[index] + n**2

    strongest_frame = sum_of_squares.index(max(sum_of_squares))
    hz = mfcc.mel2hz(features[strongest_frame])

    max_hz = max(hz)
    min_hz = min(hz)

    speech_booster = AudioEffectsChain().lowshelf(frequency=min_hz*(-1), gain=12.0, slope=0.5)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_speech_boosted = speech_booster(audio)

    features = mfcc.mfcc(y_speech_boosted, sr, 0.025, 0.01, 16, nfilt=40, nfft=512, appendEnergy = False, winfunc=np.hamming)


    features = preprocessing.scale(features) #scaling to ensure that all values are within 0 and 1

    return features

#path to training data
source   = "/Users/angelhuang/Desktop/MIT 2017-2018/2018 Summer/Via Technologies Internship/pygender/train_data/AudioSet/female"   

#path to save trained model
dest     = "/Users/angelhuang/Desktop/MIT 2017-2018/2018 Summer/Via Technologies Internship/pygender/"         

files    = [os.path.join(source,f) for f in os.listdir(source) if 
             f.endswith('.wav')] 

features = np.asarray(());

for f in files:
    sr,audio = read(f)
    vector   = get_MFCC(sr,audio)
    print(vector.shape)
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
print('finish files')

gmm = GMM(n_components = 13, n_iter = 200, covariance_type='diag', n_init = 3)
#the higher the n_iter and n_init, the longer/more accurate it will be
gmm.fit(features)

picklefile = f.split("/")[-2].split(".wav")[0]+".gmm" #creating the .gmm file
#change the "/" according to how your os writes out path (can be /, //, \\, etc)
print(picklefile)

# model saved as female.gmm
pickle.dump(gmm,open(dest+picklefile,'wb')) #writing the gmm file into folder 
print('modeling completed for gender:',picklefile)