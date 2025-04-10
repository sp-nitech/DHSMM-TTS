dataset = {
    'atr-tts-jp-corpus': {
        'datalist': 'list/train.list',
        'datadirs': {
            'phone': 'data/phn/phone_hl',
            'mel': 'data/mel/01',
        },
    },
}

meldim = 256
scaling_factor = 1.0e-5

latent_dim = 16
num_states_per_phoneme = 1
linguistic_hidden_dim = 1024
acoustic_hidden_dim = 1024

batch_size = 8
max_state_duration = 150
mel_reduction_factor = 1
em_z = False
em_s = False
em_s_enc = False
viterbi = False

bp_weighting_z = True
bp_weighting_s = True
bp_weighting_s_enc = True

max_iteration = 2000000
max_grad_norm = None
log_iter_interval = 1000
plot_iter_interval = 1000
save_iter_interval = 50000

optim = {
    'lr': 1e-4,
    'betas': (0.9, 0.98),
    'eps': 1e-9,
    'weight_decay': 0.0,
}
