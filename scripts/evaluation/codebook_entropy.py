import argbind
import audiotools as at
import numpy as np
import torch
import torchaudio
import tqdm
import json
from collections import defaultdict

from emac.utils import load_model

def _flatten_to_1d_int(x: torch.Tensor) -> torch.Tensor:
    """임의 shape의 long 텐서를 1D long으로 평탄화"""
    x = x.detach()
    if x.dtype != torch.long:
        x = x.long()
    return x.reshape(-1)

def _entropy_bits(counts: torch.Tensor) -> float:
    """counts(각 심볼 카운트)로부터 base-2 entropy (bits) 계산"""
    total = counts.sum().item()
    if total <= 0:
        return 0.0
    p = counts / counts.sum()
    p = p.clamp_min(1e-12)
    # H = -sum p log2 p
    H = -(p * p.log2()).sum().item()
    return H


def _usage_concentration(counts: torch.Tensor) -> dict:
    """Summarize how concentrated a code distribution is."""
    counts = counts.to(torch.float64)
    total = counts.sum().item()
    if total <= 0:
        return {
            "top1_ratio": 0.0,
            "top5_ratio": 0.0,
            "top10_ratio": 0.0,
            "top50_ratio": 0.0,
            "gini": 0.0,
            "unused_codes": int(counts.numel()),
            "rare_codes_le_1": 0,
            "rare_codes_le_5": 0,
            "top_codes": [],
        }

    probs = counts / total
    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    sorted_probs = sorted_counts / total
    n = int(counts.numel())
    nonzero = counts > 0

    # Gini coefficient over counts. 0 is uniform, 1 is fully concentrated.
    ascending = torch.sort(counts)[0]
    index = torch.arange(1, n + 1, dtype=torch.float64, device=counts.device)
    gini = ((2 * index - n - 1) * ascending).sum().item() / (n * total)

    def top_ratio(k: int) -> float:
        return sorted_probs[: min(k, n)].sum().item()

    top_n = min(10, n)
    top_codes = [
        {
            "code": int(sorted_indices[i].item()),
            "count": int(sorted_counts[i].item()),
            "ratio": float(sorted_probs[i].item()),
        }
        for i in range(top_n)
    ]
    return {
        "top1_ratio": top_ratio(1),
        "top5_ratio": top_ratio(5),
        "top10_ratio": top_ratio(10),
        "top50_ratio": top_ratio(50),
        "gini": float(gini),
        "unused_codes": int((counts == 0).sum().item()),
        "rare_codes_le_1": int((counts <= 1).sum().item()),
        "rare_codes_le_5": int((counts <= 5).sum().item()),
        "top_codes": top_codes,
    }

def _infer_codebook_size(model, fallback: int) -> int:
    """Prefer the loaded model's actual codebook size over CLI fallback."""
    for obj in (model, getattr(model, "quantizer", None)):
        size = getattr(obj, "codebook_size", None)
        if size is not None:
            return int(size)
    quantizers = getattr(getattr(model, "quantizer", None), "quantizers", None)
    if quantizers is not None and len(quantizers) > 0:
        size = getattr(quantizers[0], "codebook_size", None)
        if size is not None:
            return int(size)
        codebook = getattr(quantizers[0], "codebook", None)
        weight = getattr(codebook, "weight", None)
        if weight is not None:
            return int(weight.shape[0])
    return int(fallback)


def _codebook_labels(model, n_codebooks: int, scale: str, is_wave: str):
    """Return stable labels for each codebook, using model metadata when possible."""
    scale_factor = getattr(model, "scale_factor", None)
    multi_scale = getattr(model, "multi_scale", None)
    use_wavescale = bool(getattr(model, "use_wavescale", False))

    if scale_factor is not None:
        base = [float(s) for s in scale_factor]
        labels = base[::-1] + base[1:] if use_wavescale else base
    elif multi_scale is not None:
        base = [float(s) for s in multi_scale]
        max_scale = max(base) if base else 1.0
        labels = [s / max_scale for s in base]
    elif scale:
        base = [float(s) for s in scale.split(",")]
        labels = base[::-1] + base[1:] if is_wave == "True" else base
    else:
        labels = []

    out = []
    for i in range(n_codebooks):
        if i < len(labels):
            out.append(f"{labels[i]}_{i}")
        else:
            out.append(str(i))
    return out


def _label_scale_weight(label: str) -> float:
    """Best-effort token-rate weight from labels like '0.58_4'."""
    try:
        return float(str(label).split("_", 1)[0])
    except (TypeError, ValueError):
        return 1.0


def _load_audio_excerpt(path, sample_rate: int, duration: float, crop_mode: str, device):
    if crop_mode == "salient":
        sig = at.AudioSignal.salient_excerpt(path, loudness_cutoff=-20, duration=duration)
        sig = sig.resample(sample_rate).to(device)
        return sig.audio_data.mean(dim=1, keepdim=True)

    info = torchaudio.info(str(path))
    src_sr = int(info.sample_rate)
    n_src = int(round(duration * src_sr))
    if crop_mode == "center":
        offset = max(0, (int(info.num_frames) - n_src) // 2)
    elif crop_mode == "first":
        offset = 0
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    audio, sr = torchaudio.load(str(path), frame_offset=offset, num_frames=n_src)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.unsqueeze(0).to(device)

@argbind.bind(without_prefix=True, positional=True)
def main(
    folder: str,
    model_path: str,
    n_samples: int = 1024,
    device: str = "cuda",
    scale: str = "",
    is_wave: str = "",
    codebook_size : int = 1024,
    save_json: str = "",  # 결과를 JSON으로 저장하고 싶으면 경로 지정,
    duration: float = 1.0,
    crop_mode: str = "salient",
):
    # 입력 파일 수집
    files = at.util.find_audio(folder)[:n_samples]
    if len(files) == 0:
        raise RuntimeError(f"No audio files found under {folder}")

    # 모델 로드
    with torch.no_grad():
        # load_model() includes checkpoint compatibility fixes, e.g. old
        # Wavescale checkpoints with missing use_wavescale metadata.
        model = load_model(load_path=model_path).to(device)
        model.eval()

        actual_codebook_size = _infer_codebook_size(model, codebook_size)
        print(f"[INFO] codebook_size={actual_codebook_size} (fallback arg was {codebook_size})")

        # 코드북별 카운트 누적 버킷
        # key: codebook index(int) 또는 제공된 scale label, value: torch.Tensor counts (길이 K 혹은 관측 최대 idx+1)
        aggregated_counts = defaultdict(lambda: None)
        total_tokens = defaultdict(int)
        labels = None

        # 전체 샘플 인코딩
        for f in tqdm.tqdm(files, desc="Encode audio excerpts"):
            # DAC/EMAC tokenizer path: mono -> model sample-rate -> model preprocess.
            audio = _load_audio_excerpt(
                f,
                sample_rate=model.sample_rate,
                duration=duration,
                crop_mode=crop_mode,
                device=model.device,
            )
            x = model.preprocess(audio, sample_rate=model.sample_rate)
            _, codes, *_ = model.encode(x)  # codes: List[Tensor] (각 코드북별 인덱스들)

            # 코드북 수
            n_codebooks = len(codes)
            if labels is None:
                labels = _codebook_labels(model, n_codebooks, scale, is_wave)
                print("[INFO] codebook labels:", labels)
            # 코드북별 인덱스 텐서를 1D long으로 펼치고 bincount
            for cb_idx, code in enumerate(codes):
                code_1d = _flatten_to_1d_int(code.cpu())
                max_idx = int(code_1d.max().item()) if code_1d.numel() > 0 else -1
                size_needed = max(actual_codebook_size, max_idx + 1)
                counts = torch.bincount(code_1d, minlength=size_needed) if size_needed > 0 else torch.zeros(0, dtype=torch.long)

                key = labels[cb_idx]
                total_tokens[key] += int(code_1d.numel())

                if aggregated_counts[key] is None:
                    aggregated_counts[key] = counts.to(torch.long)
                else:
                    # 길이가 다르면 pad
                    cur = aggregated_counts[key]
                    if counts.size(0) > cur.size(0):
                        pad = torch.zeros(counts.size(0) - cur.size(0), dtype=torch.long)
                        cur = torch.cat([cur, pad], dim=0)
                    elif cur.size(0) > counts.size(0):
                        pad = torch.zeros(cur.size(0) - counts.size(0), dtype=torch.long)
                        counts = torch.cat([counts, pad], dim=0)
                    aggregated_counts[key] = cur + counts

        if not aggregated_counts:
            raise RuntimeError("No codes were collected. Check input audio/model.")

        print(f"코드북 개수: {len(aggregated_counts)}")
        # 정렬된 키 목록(출력 안정성)
        keys_sorted = list(labels) if labels is not None else sorted(aggregated_counts.keys(), key=str)

        per_codebook = []
        sum_H = 0.0
        sum_logK = 0.0
        weighted_sum_H = 0.0
        weighted_sum_logK = 0.0
        token_weighted_sum_H = 0.0
        token_weighted_sum_logK = 0.0

        for key in keys_sorted:
            counts = aggregated_counts[key].to(torch.float64)
            H_bits = _entropy_bits(counts)  # base-2 entropy
            
            K = max(actual_codebook_size, int(counts.numel()))

            capacity_bits = float(np.log2(K)) if K > 0 else 0.0
            eff = (H_bits / capacity_bits) if capacity_bits > 0 else 0.0
            used_codes = int((counts > 0).sum().item())
            perplexity = float(2 ** H_bits)
            scale_weight = _label_scale_weight(key)
            num_tokens = int(total_tokens[key])

            per_codebook.append({
                "key": key,
                "scale_weight": scale_weight,
                "entropy_bits": H_bits,
                "capacity_bits": capacity_bits,
                "efficiency": eff,  # 0~1
                "perplexity": perplexity,
                "used_codes": used_codes,
                "codebook_size": int(K),
                "usage_ratio": used_codes / K if K > 0 else 0.0,
                "num_tokens": num_tokens,
                **_usage_concentration(counts),
            })

            sum_H += H_bits
            sum_logK += capacity_bits
            weighted_sum_H += scale_weight * H_bits
            weighted_sum_logK += scale_weight * capacity_bits
            token_weighted_sum_H += num_tokens * H_bits
            token_weighted_sum_logK += num_tokens * capacity_bits

        overall_eff = (sum_H / sum_logK) if sum_logK > 0 else 0.0
        scale_weighted_eff = (
            weighted_sum_H / weighted_sum_logK
            if weighted_sum_logK > 0
            else 0.0
        )
        token_weighted_eff = (
            token_weighted_sum_H / token_weighted_sum_logK
            if token_weighted_sum_logK > 0
            else 0.0
        )
        frame_rate = float(model.sample_rate) / float(getattr(model, "hop_length", 1))
        entropy_kbps = frame_rate * weighted_sum_H / 1000.0
        nominal_kbps = frame_rate * weighted_sum_logK / 1000.0
        total_seconds = len(files) * duration
        token_entropy_kbps = (
            token_weighted_sum_H / total_seconds / 1000.0
            if total_seconds > 0
            else 0.0
        )
        token_nominal_kbps = (
            token_weighted_sum_logK / total_seconds / 1000.0
            if total_seconds > 0
            else 0.0
        )

        # 출력
        print("\n=== Bitrate efficiency (DAC definition) ===")
        for row in per_codebook:
            print(f"- Codebook {row['key']}: H={row['entropy_bits']:.4f} bits, "
                  f"capacity={row['capacity_bits']:.4f} bits, "
                  f"efficiency={row['efficiency']*100:.2f}%, "
                  f"used={row['used_codes']}/{row['codebook_size']} "
                  f"({row['usage_ratio']*100:.2f}%), "
                  f"perplexity={row['perplexity']:.2f}")
        print(f"> Overall efficiency = {overall_eff*100:.2f}% "
              f"(sum H / sum capacity)")
        print(f"> Scale-weighted efficiency = {scale_weighted_eff*100:.2f}% "
              f"(sum scale*H / sum scale*capacity)")
        print(f"> Token-weighted efficiency = {token_weighted_eff*100:.2f}% "
              f"(sum tokens*H / sum tokens*capacity)")
        print(f"> Entropy bitrate ~= {entropy_kbps:.3f} kbps "
              f"vs nominal ~= {nominal_kbps:.3f} kbps")
        print(f"> Token-count entropy bitrate ~= {token_entropy_kbps:.3f} kbps "
              f"vs nominal ~= {token_nominal_kbps:.3f} kbps")

        # 저장 옵션
        if save_json:
            payload = {
                "per_codebook": per_codebook,
                "overall_efficiency": overall_eff,
                "scale_weighted_efficiency": scale_weighted_eff,
                "sum_entropy_bits": sum_H,
                "sum_capacity_bits": sum_logK,
                "weighted_sum_entropy_bits": weighted_sum_H,
                "weighted_sum_capacity_bits": weighted_sum_logK,
                "token_weighted_efficiency": token_weighted_eff,
                "token_weighted_sum_entropy_bits": token_weighted_sum_H,
                "token_weighted_sum_capacity_bits": token_weighted_sum_logK,
                "entropy_kbps": entropy_kbps,
                "nominal_kbps": nominal_kbps,
                "token_entropy_kbps": token_entropy_kbps,
                "token_nominal_kbps": token_nominal_kbps,
                "model_path": model_path,
                "folder": folder,
                "n_audio_files": len(files),
                "duration": duration,
                "crop_mode": crop_mode,
                "model_sample_rate": int(model.sample_rate),
            }
            with open(save_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[INFO] Saved to {save_json}")

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
