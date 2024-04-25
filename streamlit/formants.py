import os
import numpy as np
from scipy.io import wavfile
import parselmouth 
from parselmouth.praat import call
import matplotlib.pyplot as plt 

def get_formants(path):
    fs, x = wavfile.read(path)
    return formants_praat(x,fs)

def formants_praat(x, fs):
        f0min, f0max  = 75, 300
        sound = parselmouth.Sound(x, sampling_frequency=fs)
        # pitch = sound.to_pitch()
        # f0 = pitch.selected_array['frequency']
        formants = sound.to_formant_burg(time_step=1/fs, maximum_formant=5000)
        
        # f1_list, f2_list, f3_list, f4_list  = [], [], [], []
        f1_list, f2_list  = [], []
        for t in formants.ts():
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            # f3 = formants.get_value_at_time(3, t)
            # f4 = formants.get_value_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            # if np.isnan(f3): f3 = 0
            # if np.isnan(f4): f4 = 0
            f1_list.append(f1)
            f2_list.append(f2)
            # f3_list.append(f3)
            # f4_list.append(f4)
            
        # return f0, f1_list, f2_list, f3_list, f4_list
        # return f1_list, f2_list, f3_list, f4_list
        return f1_list, f2_list

if __name__ == "__main__":
    fs, x = wavfile.read('/Users/spencer.jensen/Desktop/university/speechsynth/my_arctic_voice/wav/arctic_a0003.wav')

    # f0, f1, f2, f3, f4 = formants_praat(x,fs)
    # f1, f2, f3, f4 = formants_praat(x,fs)
    f1, f2 = formants_praat(x,fs)

    print(f"number of frames: {len(f1)}")
    print(f"frames: {len(x)}")
    # for i in range(len(f1)):
    #     print(f1[i], f2[i])

    # plt.plot(f0,'k')
    plt.plot(f1,'b')
    plt.plot(f2,'r')
    # plt.plot(f3,'g')
    # plt.plot(f4,'m')
    # plt.legend(['f0','f1','f2','f3','f4'])
    # plt.legend(['f1','f2','f3','f4'])
    plt.legend(['f1','f2'])
    plt.grid(True)
    plt.ylabel('formants(Hz)')

    plt.show()