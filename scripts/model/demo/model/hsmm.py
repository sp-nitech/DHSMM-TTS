import torch
from torch.nn import functional as F

LZERO = -1.0E+10


def get_shape(logpo, logpd):
    size1 = list(logpo.size())
    size2 = list(logpd.size())
    size1.insert(3, 1)
    size2.insert(2, 1)
    return torch.broadcast_shapes(size1, size2)

def get_mask_from_lengths(lengths, max_length=None):
    if max_length is None:
        max_length = torch.max(lengths).item()
    ids = torch.arange(0, max_length, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1))
    return ~mask

def pad_hsmm(x, y, num_states, num_frames, value=LZERO):
    # x: [B x L x T], y: [B x L x D]
    device = x.device
    B, L, T, D = get_shape(x, y)
    tmask = get_mask_from_lengths(num_frames, T).to(device)
    smask = get_mask_from_lengths(num_states, L).to(device)
    x = x.masked_fill(torch.logical_or(tmask[:, None], smask[:, :, None]), value)
    y = y.masked_fill(smask[..., None].expand(y.size()), value)
    return x, y

def double_precision(enabled=True):
    def _cast_args(func):
        def _cast(x, to_double=True):
            if isinstance(x, torch.Tensor):
                if to_double and x.dtype == torch.float:
                    return x.type(torch.double)
                if not to_double and x.dtype == torch.double:
                    return x.type(torch.float)
                
            if isinstance(x, list):
                return list(_cast(a, to_double) for a in x)
            if isinstance(x, tuple):
                return tuple(_cast(a, to_double) for a in x)
            if isinstance(x, set):
                return set(_cast(a, to_double) for a in x)
            if isinstance(x, dict):
                return dict((k, _cast(v, to_double)) for k, v in x.items())
            return x
        
        def _check(x):
            if isinstance(x, torch.Tensor) and x.dtype == torch.double:
                return True
            if isinstance(x, list) or isinstance(x, tuple) or isinstance(x, set):
                return any(_check(a) for a in x)
            if isinstance(x, dict):
                return any(_check(a) for a in x.values())
            return False
        
        def _wrapper(*args, **kwargs):
            return_double = any([_check(args), _check(kwargs)])
            
            if enabled:
                args = _cast(args)
                kwargs = _cast(kwargs)
                
            ret = func(*args, **kwargs)
            if enabled and not return_double:
                ret = _cast(ret, to_double=False)
            return ret

        return _wrapper
    return _cast_args

def update_generalized_backward_fp(lbeta, lbeta0, logpo, logpd, end, t):
    lsum = torch.logsumexp(lbeta + logpd, dim=-1)  # [B x L]
    lbeta[..., 1:] = torch.clone(lbeta[..., :-1])
    lbeta[:, :-1, 0] = lsum[:, 1:]
    lbeta[:, -1, 0] = LZERO
    lbeta[..., 0].masked_fill_(end, 0.0)
    lbeta0[..., t] = lbeta[..., 0]
    lbeta.add_(logpo[..., t:t+1])

def update_generalized_backward_bp(grad_lbeta, grad_lbeta0, grad_logpo, grad_logpd, lbeta, logpd, end, t):
    grad_logpo[..., t].add_(grad_lbeta.sum(-1))
    grad_lbeta[..., 0].add_(grad_lbeta0[..., t])

    grad_lbeta[..., 0].masked_fill_(end, 0.0)
    grad_lsum = grad_lbeta[:, :-1, :1]  # [B x L-1 x 1]
    x = lbeta + logpd  # [B x L x D]
    grad_x = grad_lsum * F.softmax(x[:, 1:], dim=-1)  # [B x L-1 x D]
    grad_logpd[:, 1:].add_(grad_x)

    prev = grad_lbeta[..., 1:].clone()
    grad_lbeta.zero_()
    grad_lbeta[..., :-1].add_(prev)
    grad_lbeta[:, 1:].add_(grad_x)

class GeneralizedBackwardAlgorithm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logpo, logpd, num_states, num_frames, batch_sizes):
        # logpo: [B x L x T], logpd: [B x L x D]
        B, L, T, D = get_shape(logpo, logpd)
        base = logpo
        lbeta = base.new_full((B, L, D), LZERO)
        lbeta0 = base.new_zeros((B, L, T))
        lbetaD = base.new_zeros((B, L, T))

        state_indices = torch.arange(start=1, end=L+1, dtype=torch.long, device=num_states.device)
        smask = (num_states[:, None] == state_indices[None]).to(base.device)  # [B x L]
        num_frames = num_frames.clone().to(smask.device)

        for t in reversed(range(T)):
            bsize = batch_sizes[t]
            end = (num_frames[:bsize, None] == t + 1) * smask[:bsize]
            update_generalized_backward_fp(lbeta[:bsize], lbeta0[:bsize], logpo[:bsize], logpd[:bsize], end, t)
            lbetaD[:bsize, ..., t].copy_(lbeta[:bsize, ..., -1])
            
        nll = -torch.logsumexp(lbeta[:, :1] + logpd[:, :1], dim=2, keepdim=True)  # [B x 1 x 1]
        lbetaT = lbeta
        ctx.save_for_backward(logpo, logpd, lbetaT, lbetaD, smask, num_frames, batch_sizes)
        return lbeta0, nll

    @staticmethod
    def backward(ctx, grad_lbeta0, grad_nll):
        # grad_lbeta0: [B x L x T], grad_nll: [B x 1 x 1]
        logpo, logpd, lbetaT, lbetaD, smask, num_frames, batch_sizes, = ctx.saved_tensors
        B, L, T, D = get_shape(logpo, logpd)
        base = logpo
        
        grad_logpo = base.new_zeros((B, L, T))
        grad_logpd = base.new_zeros((B, L, D))
        grad_lbeta = base.new_zeros((B, L, D))
        lbeta = lbetaT  # [B x L x D]

        x = lbeta[:, :1] + logpd[:, :1]
        grad_x = grad_nll * -F.softmax(x, dim=2)
        grad_lbeta[:, :1].add_(grad_x)
        grad_logpd[:, :1].add_(grad_x)
        
        for t in range(T):
            bsize = batch_sizes[t]
            lbeta[:bsize, ..., :-1] = lbeta[:bsize, ..., 1:] - logpo[:bsize, ..., t:t+1]
            lbeta[:bsize, ..., -1] = lbetaD[:bsize, ..., t+1] if t+1 < T else base.new_full((bsize, L), LZERO)
            end = (num_frames == t + 1)[:bsize, None] * smask[:bsize]
            update_generalized_backward_bp(grad_lbeta[:bsize], grad_lbeta0[:bsize], grad_logpo[:bsize], grad_logpd[:bsize], lbeta[:bsize], logpd[:bsize], end, t)

        return grad_logpo, grad_logpd, None, None, None

def update_generalized_forward_fp(lalpha, logpo, logpd, t):
    lsum = torch.logsumexp(lalpha[:, :-1] + logpd[:, :-1], dim=-1)  # [B x L-1]
    lalpha[..., 1:] = torch.clone(lalpha[..., :-1])
    lalpha[:, 0, 0] = 0.0 if t == 0 else LZERO
    lalpha[:, 1:, 0] = lsum
    lalpha.add_(logpo[..., t:t+1])

def update_generalized_forward_bp(grad_lalpha, grad_logpo, grad_logpd, lalpha, logpd, t):
    grad_logpo[..., t].add_(grad_lalpha.sum(-1))
    
    grad_lsum = grad_lalpha[:, 1:, :1]  # [B x L-1 x 1]
    x = lalpha + logpd  # [B x L x D]
    grad_x = grad_lsum * F.softmax(x[:, :-1], dim=-1)  # [B x L-1 x D]
    grad_logpd[:, :-1].add_(grad_x)

    prev = grad_lalpha[..., 1:].clone()
    grad_lalpha.zero_()
    grad_lalpha[..., :-1].add_(prev)
    grad_lalpha[:, :-1].add_(grad_x)
    
def update_expectation_fp(xt, tgamma, dgamma, t):
    D = dgamma.size(-1)
    dgamma.add_(xt)
    dlen = t + 1 - max(0, t + 1 - D)
    cumsum_xt = xt[..., :dlen].flip(dims=(-1, )).cumsum(dim=-1)
    tgamma[..., t+1-dlen:t+1] += cumsum_xt

def update_expectation_bp(grad_x, grad_tgamma, grad_dgamma, t):
    B, L, T = grad_tgamma.size()
    D = grad_dgamma.size(2)
    dlen = t + 1 - max(0, t + 1 - D)
    grad_x.copy_(grad_dgamma)
    grad_x[..., :dlen].add_(grad_tgamma[:, :, t+1-dlen:t+1].flip(dims=(-1, )).cumsum(dim=-1))

class GeneralizedForwardAlgorithm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logpo, logpd, lbeta0, nll, batch_sizes):
        # logpo: [B x L x T], logpd: [B x L x D], lbeta0: [B x L x T], nll: [B x 1 x 1]
        B, L, T, D = get_shape(logpo, logpd)
        base = logpo
        tgamma = base.new_zeros((B, L, T))
        dgamma = base.new_zeros((B, L, D))
        
        lalpha = base.new_full((B, L, D), LZERO)
        lalphaD = base.new_zeros((B, L, T))
        
        c = logpd + nll
        for t in range(T):
            bsize = batch_sizes[t]
            update_generalized_forward_fp(lalpha[:bsize], logpo[:bsize], logpd[:bsize], t)
            lalphaD[:bsize, ..., t].copy_(lalpha[:bsize, ..., -1])
            xt = torch.exp(lalpha[:bsize] + lbeta0[:bsize, ..., t:t+1] + c[:bsize])  # [B x L x D]
            update_expectation_fp(xt, tgamma[:bsize], dgamma[:bsize], t)

        lalphaT = lalpha
        ctx.save_for_backward(logpo, logpd, lalphaT, lalphaD, lbeta0, nll, batch_sizes)
        return tgamma, dgamma

    @staticmethod
    def backward(ctx, grad_tgamma, grad_dgamma):
        # grad_tgamma: [B x L x T], grad_dgamma: [B x L x D]
        logpo, logpd, lalphaT, lalphaD, lbeta0, nll, batch_sizes, = ctx.saved_tensors
        B, L, T, D = get_shape(logpo, logpd)
        
        base = grad_tgamma
        grad_logpo = base.new_zeros((B, L, T))
        grad_logpd = base.new_zeros((B, L, D))
        grad_lbeta0 = base.new_zeros((B, L, T))
        grad_nll = base.new_zeros((B, 1, 1))
        
        grad_x = base.new_zeros((B, L, D))
        grad_lalpha = base.new_zeros((B, L, D))
        
        c = logpd + nll
        lalpha = lalphaT
        for t in reversed(range(T)):
            bsize = batch_sizes[t]
            update_expectation_bp(grad_x[:bsize], grad_tgamma[:bsize], grad_dgamma[:bsize], t)
            xt = torch.exp(lalpha[:bsize] + lbeta0[:bsize, ..., t:t+1] + c[:bsize])
            grad_logxt = xt.mul_(grad_x[:bsize])  # [B x L x D]
            
            lalpha[:bsize, ..., :-1] = lalpha[:bsize, ..., 1:] - logpo[:bsize, ..., t:t+1]
            lalpha[:bsize, ..., -1] = lalphaD[:bsize, ..., t-1] if 0 < t else base.new_full((bsize, L), LZERO)
            
            grad_lalpha[:bsize].add_(grad_logxt)
            grad_lbeta0[:bsize, ..., t].add_(grad_logxt.sum(-1))
            grad_logpd[:bsize].add_(grad_logxt)
            grad_nll[:bsize].add_(grad_logxt.sum(dim=(1, 2), keepdims=True))
            update_generalized_forward_bp(grad_lalpha[:bsize], grad_logpo[:bsize], grad_logpd[:bsize], lalpha[:bsize], logpd[:bsize], t)
            
        return grad_logpo, grad_logpd, grad_lbeta0, grad_nll, None


@double_precision(True)
def differentiable_generalized_forward_backward_algorithm(logpo, logpd, num_states, num_frames):
    B, L, T, D = get_shape(logpo, logpd)
    base = logpo
    if num_states is None: num_states = base.new_full((B, ), L, dtype=torch.long)
    if num_frames is None: num_frames = base.new_full((B, ), T, dtype=torch.long)

    assert torch.any(num_frames <= T), (
        'The specified number of frames {} is greater than the maximum number of frames {} for logP(o).'.format(num_frames.tolist(), T))
    assert torch.any(num_states <= L), (
        'The specified number of frames {} is greater than the maximum number of states {} for logP(o).'.format(num_states.tolist(), L))
    assert torch.any(num_frames <= num_states * D), (
        'Maximum duration D={} is too small. The number of states and the number of frames are {} and {}, respectively.'.format(D, num_states.tolist(), num_frames.tolist()))
    assert torch.any(num_states <= num_frames), (
        'Too many states. The number of states and the number of frames are {} and {}, respectively.'.format(num_states.tolist(), num_frames.tolist()))

    num_frames, sorted_indices = torch.sort(num_frames, descending=True)
    indices = torch.arange(start=0, end=T)
    batch_sizes = torch.sum(indices[None] < num_frames[:, None], dim=0)  # [T]
    num_states = torch.index_select(num_states, dim=0, index=sorted_indices)
    sorted_indices = sorted_indices.to(base.device)
    logpo = torch.index_select(logpo, dim=0, index=sorted_indices)
    logpd = torch.index_select(logpd, dim=0, index=sorted_indices)

    lbeta0, nll = GeneralizedBackwardAlgorithm.apply(logpo, logpd, num_states, num_frames, batch_sizes)
    tgamma, dgamma = GeneralizedForwardAlgorithm.apply(logpo, logpd, lbeta0, nll, batch_sizes)
    tgamma, dgamma = pad_hsmm(tgamma, dgamma, num_states, num_frames, value=0.0)

    unsorted_indices = sorted_indices.argsort()
    tgamma = torch.index_select(tgamma, dim=0, index=unsorted_indices)
    dgamma = torch.index_select(dgamma, dim=0, index=unsorted_indices)
    nll = torch.index_select(nll, dim=0, index=unsorted_indices)
    return tgamma, dgamma, nll


@torch.no_grad()
@double_precision(True)
def viterbi(logpo, logpd, num_states, num_frames):
    B, L, T, D = get_shape(logpo, logpd)

    base = logpo
    lalpha = base.new_full((B, L, D), LZERO)
    bp = base.new_zeros((B, L, T+1), dtype=torch.long)
    for t in range(T):
        lmax, idx = torch.max(lalpha + logpd, dim=-1)  # [B x L]
        lalpha[..., 1:] = lalpha[..., :-1].clone()
        lalpha[:, 0, 0] = 0.0 if t == 0 else LZERO
        lalpha[:, 1:, 0] = lmax[:, :-1]
        lalpha += logpo[..., t:t+1]
        bp[..., t] = idx + 1
    else:
        bp[..., -1] = torch.max(lalpha + logpd, dim=-1)[1] + 1
    bp = bp[..., 1:]  # [B x L x T]

    frame_indices = num_frames.data.clone().to(base.device) - 1  # [B]
    num_states = num_states.data.clone().to(base.device)  # [B x L]
    dur = base.new_zeros((B, L), dtype=torch.long)
    for state_index in reversed(range(L)):
        d = torch.index_select(bp[:, state_index], dim=-1, index=frame_indices).diag()  # [B]
        d.masked_fill_(num_states <= state_index, 0)
        dur[:, state_index] = d
        frame_indices -= d
    assert torch.sum(dur.sum(dim=1).to(num_frames.device) == num_frames).item() == B
    return dur
