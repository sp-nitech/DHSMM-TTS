import math
import torch
from torch import nn
from contextlib import nullcontext

from .layers import LSTM
from .hsmm import differentiable_generalized_forward_backward_algorithm as forward_backward
from .hsmm import viterbi
from .hsmm import pad_hsmm
from .gaussian import reparameterization_trick
from .gaussian import log_prob1d, log_prob2d, extended_log_prob2d, kld1d, kld2d
from .gaussian import gaussian_approximation, multiply_2gaussians, gaussian_bp_weighting
from phone import phone_label_list
import config

LOG2PI = 1.8378770664093453
LZERO = -1.0E+10

from .plot import plot
plot_items = dict()

def split_gauss_param(x):
    mean, logvar = x.chunk(2, dim=-1)
    return (mean, logvar, )

def get_mask_from_lengths(lengths, max_length=None):
    if max_length is None:
        max_length = torch.max(lengths).item()
    ids = torch.arange(0, max_length, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1))
    return ~mask

def mask_padding(x, lengths, max_length=None):
    mask = get_mask_from_lengths(lengths, max_length=max_length).to(x.device)
    if x.ndim == 3:
        mask = mask[..., None].expand(x.size())
    return x.masked_fill(mask, 0.0)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_states):
        super().__init__()
        self.num_states = num_states
        assert 1 <= num_states
        num_symbols = len(phone_label_list)
        embedding_dim = input_dim
        lstm_dim = hidden_dim
        self.embedding = nn.Embedding(num_symbols, embedding_dim)
        if 1 < num_states:
            state_dim = 64
            self.state_embedding = nn.Embedding(num_states, state_dim)
        else:
            state_dim = 0
            self.state_embedding = None
        self.lstm = LSTM(embedding_dim + state_dim, lstm_dim // 2, 2, batch_first=True, bidirectional=True)
        
    def forward(self, x, lengths=None):
        e = self.embedding(x)
        if self.state_embedding is not None:
            B, L, _ = e.size()
            e = torch.repeat_interleave(e, repeats=self.num_states, dim=1)
            state = self.state_embedding.weight.repeat(L, 1).expand(B, L * self.num_states, -1)
            e = torch.cat([e, state], dim=-1)
            if lengths is not None:
                lengths = lengths * self.num_states
        y = self.lstm(e, lengths)
        return y, lengths
    
class DurationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2))
        
    def forward(self, x):
        mean, logvar = split_gauss_param(self.ffnn(x))
        return (mean, logvar.clamp(min=math.log(0.01 ** 2)))

class OutputModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2 * output_dim))
        
    def forward(self, x):
        mean, logvar = split_gauss_param(self.ffnn(x))
        return (mean, logvar.clamp(min=math.log(0.01 ** 2)))
    
class SpeechEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        lstm_dim = hidden_dim
        self.lstm = LSTM(input_dim, lstm_dim // 2, num_layers=2,
                         batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_dim, 2 * output_dim)
        
    def forward(self, x, lengths=None):
        y = self.lstm(x, lengths)
        dist = split_gauss_param(self.proj(y))
        return dist

class SpeechDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        lstm_dim = hidden_dim
        self.lstm = LSTM(input_dim, lstm_dim // 2, num_layers=2,
                         batch_first=True, bidirectional=True)
        self.projection = nn.Linear(lstm_dim, 2 * output_dim)
    
    def forward(self, z, lengths):
        x = self.lstm(z, lengths)
        dist = split_gauss_param(self.projection(x))
        return dist

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_state_dur = config.max_state_duration // config.mel_reduction_factor
        self.outdim = config.meldim * config.mel_reduction_factor

        latent_dim = config.latent_dim
        embedding_dim = config.linguistic_hidden_dim
        context_dim = config.linguistic_hidden_dim
        hidden_dim1 = config.linguistic_hidden_dim
        hidden_dim2 = config.acoustic_hidden_dim
        
        self.text_encoder = TextEncoder(embedding_dim, context_dim, hidden_dim=hidden_dim1, num_states=config.num_states_per_phoneme)
        self.durmodel = DurationModel(context_dim, hidden_dim=hidden_dim1)
        self.outmodel = OutputModel(context_dim, latent_dim, hidden_dim=hidden_dim1)
        self.speech_encoder = SpeechEncoder(self.outdim, latent_dim, hidden_dim=hidden_dim2)
        self.speech_decoder = SpeechDecoder(latent_dim, self.outdim, hidden_dim=hidden_dim2)
        
        self.register_buffer('duration', torch.arange(start=1, end=self.max_state_dur+1, dtype=torch.float).reshape(1, -1, 1))
        
    def forward_mdn(self, inputs, lengths):
        
        ctx, state_lengths = self.text_encoder(inputs, lengths)
        ps = self.durmodel(ctx)
        po_s = self.outmodel(ctx)
        
        return po_s, ps, state_lengths

    @staticmethod
    def calculate_logpo(lo, po_s, pw):

        if config.em_s_enc:
            lo = tuple(x.detach() for x in lo)
        elif config.bp_weighting_s_enc:
            lo = gaussian_bp_weighting(*lo, pw)
        
        if config.em_s:
            po_s = tuple(x.detach() for x in po_s)
        elif config.bp_weighting_s:
            po_s = gaussian_bp_weighting(*po_s, pw)
            
        lmu, llogvar = lo    # [B x T x C]            
        pmu, plogvar = po_s  # [B x L x C]

        logpw = math.log(pw) if 0.0 < pw else LZERO
        C = lmu.size(-1)
        logcoef = -0.5 * ((pw - 1.0) * LOG2PI * C + logpw * C + (pw - 1.0) * torch.sum(plogvar, dim=-1, keepdim=True))
        logpo = logcoef + extended_log_prob2d(lmu, llogvar, pmu, plogvar - logpw)  # [B x L x T]

        return logpo

    @staticmethod
    def calculate_logpd(d, ps, pw):
        
        if config.em_s:
            ps = tuple(x.detach() for x in ps)
        elif config.bp_weighting_s:
            ps = gaussian_bp_weighting(*ps, pw)
        
        logpd = pw * log_prob2d(d, *ps)  # [B x L x D]
        
        return logpd
    
    def estimate_alignment(self, lo, ps, po_s, num_states, num_frames, pw):

        if config.em_s and config.em_s_enc:
            ctx = torch.no_grad() 
        else:
            ctx = nullcontext()
        
        with ctx:
        
            logpo = self.calculate_logpo(lo, po_s, pw)
            logpd = self.calculate_logpd(self.duration, ps, pw)
            logpo, logpd = pad_hsmm(logpo, logpd, num_states, num_frames)

            if config.viterbi:
                assert config.em_s
                d = viterbi(logpo, logpd, num_states, num_frames)
                cumd = torch.cumsum(torch.cat([d.new_ones(d.size(0), 1), d], dim=1), dim=1)[..., None]  # [B x L+1 x 1]
                begin = cumd[:, :-1]
                end = cumd[:, 1:]
                t = torch.arange(1, cumd.max().item(), device=d.device).reshape(1, 1, -1)  # [1 x 1 x T]
                tgamma = torch.logical_and(begin <= t, t < end).float()
                dgamma = (self.duration.transpose(1, 2).long() == d[..., None]).float()
                entropy = tgamma.new_zeros((d.size(0), ))
            else:
                tgamma, dgamma, nll = forward_backward(logpo, logpd, num_states, num_frames)
                assert nll.size(1) == nll.size(2) == 1
                nll = nll[:, 0, 0]
                entropy = -(nll + torch.sum(tgamma * logpo, dim=(1, 2)) + torch.sum(dgamma * logpd, dim=(1, 2)))

        global plot_items
        plot_items['logpo'] = logpo
        plot_items['logpd'] = logpd
        
        return tgamma, dgamma, entropy
    
    @staticmethod
    def approximate_posterior_s(lo, po_s, tgamma, pw):
        
        if config.em_z:
            po_s = tuple(x.detach() for x in po_s)
        elif config.bp_weighting_z:
            po_s = gaussian_bp_weighting(*po_s, pw)

        logpw = math.log(pw) if 0.0 < pw else LZERO
        po_s = (po_s[0], po_s[1] - logpw, )

        ngamma = tgamma / (tgamma.sum(dim=2, keepdim=True) + 1e-8)  # [B x L x T]
        lo_s = gaussian_approximation(ngamma, *lo)  # [B x L x C]
        
        qo_s = multiply_2gaussians(*lo_s, *po_s)  # [B x L x C]
        
        return qo_s

    @staticmethod
    def approximate_posterior_o(lo, qo_s, tgamma, pw):

        qo = gaussian_approximation(tgamma.mT, *qo_s)  # [B x T x C]

        return qo
    
    def forward(self, inputs, targets, lengths, pw, writer):

        assert 0.0 <= pw
        batch_size = inputs.size(0)
        input_lengths, target_lengths = lengths
        
        po_s, ps, state_lengths = self.forward_mdn(inputs, input_lengths)

        lo = self.speech_encoder(targets, target_lengths)

        tgamma, dgamma, sentropy = self.estimate_alignment(lo, ps, po_s, state_lengths, target_lengths, pw)
        sloss = - torch.sum(dgamma * log_prob2d(self.duration, *ps)) - sentropy.sum()

        po = gaussian_approximation(tgamma.mT, *po_s)  
        
        qo_s = self.approximate_posterior_s(lo, po_s, tgamma, pw)
        zloss_s = torch.sum(mask_padding(tgamma.sum(-1) * kld1d(*qo_s, *po_s), state_lengths))

        qo = self.approximate_posterior_o(lo, qo_s, tgamma, pw)
        zloss_o = torch.sum(tgamma * kld2d(*qo, *qo_s))

        px = self.speech_decoder(reparameterization_trick(*qo), target_lengths)
        oloss = - mask_padding(log_prob1d(targets, *px), target_lengths).sum()

        loss = oloss + zloss_o + zloss_s + sloss 
        loss = loss * config.scaling_factor / batch_size
        
        writer('loss', loss)
        writer('oloss', oloss)
        writer('zloss_o', zloss_o)
        writer('zloss_s', zloss_s)
        writer('sloss', sloss)
        writer('pw', pw)

        No1 = state_lengths.sum() * po_s[1].size(-1)
        No2 = target_lengths.sum() * lo[1].size(-1)
        Nx = target_lengths.sum() * px[1].size(-1)
        writer('logvar/po_s max', mask_padding(po_s[1], state_lengths).max())
        writer('logvar/po_s', mask_padding(po_s[1], state_lengths).sum() / No1)
        writer('logvar/lo', mask_padding(lo[1], target_lengths).sum() / No2)
        writer('logvar/lo max', mask_padding(lo[1], target_lengths).max())
        writer('logvar/qo', mask_padding(qo[1], target_lengths).sum() / No2)
        writer('logvar/px', mask_padding(px[1], target_lengths).sum() / Nx)

        global plot_items
        plot_items['inputs'] = inputs
        plot_items['targets'] = targets
        plot_items['preds'] = px[0]
        plot_items['tgamma'] = tgamma
        plot_items['dgamma'] = dgamma
        plot_items['lo_mean'] = lo[0]
        plot_items['lo_logvar'] = lo[1]
        plot_items['qo_mean'] = qo[0]
        plot_items['qo_logvar'] = qo[1]
        plot_items['po_mean'] = po[0]
        plot_items['po_logvar'] = po[1]
        
        plot(writer, lengths, **plot_items)
        plot_items = dict()
        
        return loss
