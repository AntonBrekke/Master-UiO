import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

A1 = 1.0
A2 = 1.6
f1 = 125
f2 = 200
t1 = 0.25
t2 = 1.0
sigma1 = 0.05
sigma2 = 0.1

fs = 1e3
T = 1.5
N = int(T*fs)

t = np.linspace(0, T, N)

x_n = A1*np.sin(2*np.pi*f1*t)*np.exp(-(t-t1)**2 / sigma1**2) + A2*np.sin(2*np.pi*f2*t)*np.exp(-(t-t2)**2 / sigma2**2)

def STFT(x_n, f_samp, overlapp, vindu_størrelse, filter=False):

    t_maks = len(x_n)/f_samp # sekunder
    punkter_vindu = int((vindu_størrelse*f_samp))
    
    # Antall frekvenser gitt av tidsampling // 2 pga "speiling" 
    antall_frekvenser = int((punkter_vindu//2))

    # Med overlapp
    punkter_ikke_overlappende = int(punkter_vindu*(1-overlapp))
    totalt_antall_vindu = int(len(x_n)/punkter_ikke_overlappende)

    # resultat matrise: antall tider mot frekvens
    result = np.zeros((antall_frekvenser, totalt_antall_vindu)) 

    if filter == False:
        # Ingen vindu-funksjon
        vindu_funksjon = np.ones(punkter_vindu) 
    else:
        # Hanningvindu - en halv cosinus
        vindu_funksjon = signal.windows.hann(punkter_vindu) 

    # må legge til nuller på slutten av dataene for at
    # STFT skal fungere for det siste viduet. 
    x_n_pad = np.concatenate((x_n, np.zeros(punkter_vindu))) 

    for i in range(totalt_antall_vindu):
        # Plukk ut signalbiten
        start_index = int(punkter_ikke_overlappende*i)
        slutt_index = int(start_index + punkter_vindu)
        signalbit = x_n_pad[start_index:slutt_index]
        
        final_signalbit = signalbit*vindu_funksjon

        # FFT av signalbiten
        signalbit_FT = np.fft.fft(final_signalbit)/(punkter_vindu)

        # Lagre det fourertransformerte signalet i resultatmatrisen.
        # Vi er kun intereset i resultat for positive frekvenser.
        # Ganger derfor med 2 pga. "speiling"
        result[:, i] = np.abs(signalbit_FT[:antall_frekvenser])*2 

    # Lager en tidsakse: skaleres med og uten overlapp 
    if overlapp == 0:
        t = np.linspace(vindu_størrelse*(0.5), t_maks - vindu_størrelse*.5, totalt_antall_vindu)
    else:
        t = np.linspace(vindu_størrelse*(1/3), t_maks + vindu_størrelse*(1/3), totalt_antall_vindu)
    
    
    # lag en frevkensarray på "vanlig måte" for FFT
    f_FFT = np.fft.fftfreq((punkter_vindu), 1/f_samp)
    # vi er kun intereset i positive frekvenser
    f = f_FFT[:antall_frekvenser]
    
    return [t, f, result]

overlapp = 0.5
vindu_størrelse = 0.1

t, f, result = STFT(x_n, fs, overlapp, vindu_størrelse, filter=True)

plt.pcolormesh(t, f, result, shading='auto')
plt.colorbar()
plt.show()