import torch
import math

LOG2PI = 1.8378770664093453

def reparameterization_trick(mean, logvar):
    return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

def log_prob1d(target, mean, logvar, dim=-1):
    C = target.size(dim)
    out = -0.5 * (LOG2PI * C + logvar.sum(dim=dim) + torch.sum(torch.pow(target - mean, 2.0) * torch.exp(-logvar), dim=dim))
    return out

def log_prob2d(target, mean, logvar):
    # target: [B x L1 x C], mean/logvar: [B x L2 x C] -> [B x L2 x L1]
    target = target.mT  # [B x C x L1]
    dim1, dim2 = -2, -1
    C = target.size(dim1)
    invvar = torch.exp(-logvar)
    mahalanobis = invvar @ target.pow(2.0) - 2.0 * (mean * invvar) @ target + torch.sum(mean.pow(2.0) * invvar, dim=dim2, keepdim=True)
    out = -0.5 * (LOG2PI * C + logvar.sum(dim=dim2, keepdim=True) + mahalanobis)
    return out

class ExtendedLogProbabilityDensityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean1, logvar1, mean2, logvar2):
        mean1 = mean1.mT  # [B x C x L1]
        logvar1 = logvar1.mT  # [B x C x L1]
        B, C, L1 = mean1.size()
        B, L2, _ = mean2.size()
        
        y = mean1.new_zeros((B, L2, L1))
        for i in range(C):
            logvar = torch.logaddexp(logvar1[:, i:i+1], logvar2[..., i:i+1])  # [B x L2 x L1]
            y.add_(logvar)
            invvar = logvar.neg_().exp_()
            diff = mean1[:, i:i+1] - mean2[..., i:i+1]
            y.add_(diff.pow_(2.0).mul_(invvar))
        y.add_(LOG2PI * C).mul_(-0.5)
        
        ctx.save_for_backward(mean1, logvar1, mean2, logvar2)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        mean1, logvar1, mean2, logvar2, = ctx.saved_tensors
        B, C, L1 = mean1.size()
        B, L2, _ = mean2.size()

        grad_mean1 = grad_y.new_empty((B, L1, C))
        grad_logvar1 = grad_y.new_empty((B, L1, C))
        grad_mean2 = grad_y.new_empty((B, L2, C))
        grad_logvar2 = grad_y.new_empty((B, L2, C))
        
        coef1 = grad_y.new_empty((B, L2, L1))
        coef2 = grad_y.new_empty((B, L2, L1))
        for i in range(C):
            lv1 = logvar1[:, i:i+1]
            lv2 = logvar2[..., i:i+1]
            nlogvar = torch.logaddexp(lv1, lv2).neg_()  # [B x L2 x L1]
            coef1.copy_(nlogvar).add_(lv1).exp_()
            coef2.copy_(nlogvar).add_(lv2).exp_()
            invvar = nlogvar.exp_()
            
            diff = mean1[:, i:i+1] - mean2[..., i:i+1]
            grad = diff.pow(2.0)
            grad.mul_(invvar).sub_(1.0).mul_(0.5).mul_(grad_y)
            grad_logvar1[..., i] = torch.sum(coef1.mul_(grad), dim=1)
            grad_logvar2[..., i] = torch.sum(coef2.mul_(grad), dim=2)
            
            diff.mul_(invvar)
            grad = diff.mul_(grad_y)  # [B x L2 x L1]
            grad_mean1[..., i] = -grad.sum(1)
            grad_mean2[..., i] = grad.sum(2)
        return grad_mean1, grad_logvar1, grad_mean2, grad_logvar2
extended_log_prob2d = ExtendedLogProbabilityDensityFunction.apply

def cross_entropy1d(post_mean, post_logvar, prior_mean, prior_logvar, dim=-1):
    C = post_mean.size(dim)
    mahalanobis = torch.sum(torch.pow(post_mean - prior_mean, 2.0) * torch.exp(-prior_logvar), dim=dim)
    ce = 0.5 * (LOG2PI * C + prior_logvar.sum(dim=dim) + mahalanobis + torch.exp(post_logvar - prior_logvar).sum(dim=dim))
    return ce

def cross_entropy2d(post_mean, post_logvar, prior_mean, prior_logvar):
    post_mean = post_mean.mT  # [B x C x L1]
    post_logvar = post_logvar.mT
    dimQ, dimP = -2, -1
    C = post_mean.size(dimQ)
    invvar = torch.exp(-prior_logvar)  # [B x L2 x C]
    mahalanobis = invvar @ post_mean.pow(2.0) - 2.0 * (prior_mean * invvar) @ post_mean + torch.sum(prior_mean.pow(2.0) * invvar, dim=dimP, keepdim=True)
    ce = 0.5 * (LOG2PI * C + prior_logvar.sum(dim=dimP, keepdim=True) + mahalanobis + invvar @ post_logvar.exp())
    return ce  # [B x L2 x L1]

def entropy1d(mean, logvar, dim=-1):
    C = mean.size(dim)
    e = 0.5 * ((LOG2PI + 1.0) * C + logvar.sum(dim=dim))
    return e

def entropy2d(_, logvar):
    logvar = logvar.mT
    dimQ = -2
    C = logvar.size(dimQ)
    e = 0.5 * ((LOG2PI + 1.0) * C + logvar.sum(dim=dimQ, keepdim=True))
    return e

def kld1d(post_mean, post_logvar, prior_mean, prior_logvar, dim=-1):
    return cross_entropy1d(post_mean, post_logvar, prior_mean, prior_logvar, dim=dim) - entropy1d(post_mean, post_logvar, dim=dim)

def kld2d(post_mean, post_logvar, prior_mean, prior_logvar):
    return cross_entropy2d(post_mean, post_logvar, prior_mean, prior_logvar) - entropy2d(post_mean, post_logvar)

def gaussian_approximation(pi, mean, logvar):
    invvar = torch.exp(-logvar)  # [B x T x C]
    denom = pi @ invvar  # [B x L x C]
    numer = pi @ (mean * invvar)  # [B x L x C]
    new_var = 1.0 / denom.clamp(min=1e-8)
    new_logvar = torch.log(new_var)
    new_mean = new_var * numer
    return new_mean, new_logvar

def multiply_2gaussians(mean1, logvar1, mean2, logvar2):
    nlogvar1, nlogvar2 = -logvar1, -logvar2
    new_logvar = -torch.logaddexp(nlogvar1, nlogvar2)
    log_coef1 = nlogvar1 + new_logvar
    log_coef2 = nlogvar2 + new_logvar
    new_mean = log_coef1.exp() * mean1 + log_coef2.exp() * mean2
    return new_mean, new_logvar


class GaussianBPWeightingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, logvar, bwt):
        ctx.bwt=bwt
        return mean, logvar
    @staticmethod
    def backward(ctx, grad_mean, grad_logvar):
        bwt = ctx.bwt
        return grad_mean*bwt, grad_logvar*bwt, None

gaussian_bp_weighting = GaussianBPWeightingFunction.apply


