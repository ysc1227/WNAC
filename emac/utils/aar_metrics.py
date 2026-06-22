import torch
from audiotools import AudioSignal
from panns_inference import AudioTagging

@torch.no_grad()
def gaussian_kl_clap(real_feats: torch.Tensor,
                     gen_feats: torch.Tensor,
                     eps: float = 1e-4) -> torch.Tensor:
    """
    real_feats: [N_r, D] CLAP embedding for real audio
    gen_feats : [N_g, D] CLAP embedding for generated audio
    eps       : diagonal regularization for covariances

    return: scalar tensor, KL(N_real || N_gen)
    """
    assert real_feats.dim() == 2 and gen_feats.dim() == 2, \
        f"real_feats {real_feats.shape}, gen_feats {gen_feats.shape} 둘 다 [N, D]여야 함"

    # 둘 다 같은 디바이스/타입으로
    device = real_feats.device
    real_feats = real_feats.to(device=device, dtype=torch.float64)
    gen_feats  = gen_feats.to(device=device, dtype=torch.float64)

    N_r, D = real_feats.shape
    N_g, D2 = gen_feats.shape
    assert D == D2, "real / gen embedding dim이 다름"

    # 1) mean
    mu_r = real_feats.mean(dim=0)          # [D]
    mu_g = gen_feats.mean(dim=0)           # [D]

    # 2) centered
    Xr = real_feats - mu_r
    Xg = gen_feats  - mu_g

    # 3) sample covariance (unbiased)
    # cov = E[(x - mu)(x - mu)^T]
    cov_r = (Xr.t() @ Xr) / (N_r - 1)      # [D, D]
    cov_g = (Xg.t() @ Xg) / (N_g - 1)      # [D, D]

    # 4) 수치 안정화: 대칭화 + eps * I로 SPD 보정
    I = torch.eye(D, dtype=torch.float64, device=device)
    cov_r = 0.5 * (cov_r + cov_r.t()) + eps * I
    cov_g = 0.5 * (cov_g + cov_g.t()) + eps * I

    # 5) Cholesky로 Σ_g^{-1}, log det Σ_r, log det Σ_g 계산
    #    (직접 inverse / det 쓰는 것보다 안정적)
    chol_g = torch.linalg.cholesky(cov_g)             # [D, D]
    inv_cov_g = torch.cholesky_inverse(chol_g)        # Σ_g^{-1}
    logdet_cov_g = 2.0 * torch.log(torch.diag(chol_g)).sum()

    chol_r = torch.linalg.cholesky(cov_r)
    logdet_cov_r = 2.0 * torch.log(torch.diag(chol_r)).sum()

    # 6) trace term: tr(Σ_g^{-1} Σ_r)
    trace_term = torch.trace(inv_cov_g @ cov_r)

    # 7) quadratic term: (μ_g - μ_r)^T Σ_g^{-1} (μ_g - μ_r)
    diff = (mu_g - mu_r).unsqueeze(0)      # [1, D]
    quad_term = (diff @ inv_cov_g @ diff.t()).squeeze()  # scalar

    # 8) log det term
    logdet_term = logdet_cov_g - logdet_cov_r

    # 9) KL
    kl = 0.5 * (trace_term + quad_term - D + logdet_term)

    # 수치 오차로 음수 살짝 나올 수 있어서 0 밑은 잘라줌 (이론상 ≥ 0)
    kl = kl.clamp_min(0.0)
    return kl.to(dtype=torch.float32)

@torch.no_grad()
def pann_probs_from_signal(sig: AudioSignal, pann_model: torch.nn.Module, device):
    """
    sig: AudioSignal (B, 1, T) 가정
    반환: [B, K] softmax probs (AudioSet 527-class)
    """
    # PANN 기본 SR = 32000
    sig32 = sig.resample(32000)
    probs_all = []

    for b in range(sig32.batch_size):
        # (1, T) -> (T,)
        audio = sig32[b].audio_data.squeeze(0)

        pann_model.eval()
        output_dict = pann_model(audio, None)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        # 'clipwise_output': [K] sigmoid score
        clip_logits = torch.from_numpy(clipwise_output[0]).to(device)  # [K]
        # IS 용이니까 softmax 확률로 변환
        p = torch.softmax(clip_logits, dim=-1)
        probs_all.append(p.unsqueeze(0))

    probs = torch.cat(probs_all, dim=0)   # [B, K]
    return probs

@torch.no_grad()
def inception_score_from_probs(probs: torch.Tensor, eps: float = 1e-8):
    """
    probs: [N, K] (softmax 확률)
    """
    p_y = probs.mean(dim=0, keepdim=True)         # [1, K]
    kl = probs * (torch.log(probs + eps) - torch.log(p_y + eps))
    kl = kl.sum(dim=1).mean()
    return torch.exp(kl)