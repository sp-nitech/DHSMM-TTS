# speech analysis conditions
sr = 48000  # sampling rate (Hz)
fs = 600    # frame shift (point)
wl = 2400   # window length (point)
ft = 4096   # fft length (point)

feature_types = ('mel', )  # ('mel', 'lienar')
mfmin = 60     # lowest frequency into Mel-frequency bins (Hz)
mfmax = 24000   # highest frequency into Mel-frequency bins (Hz)
num_mels = 256  # number of Mel bands
