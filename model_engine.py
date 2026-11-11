import io, json, os, time
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Mel Filterbank & Feature Extraction ---
def htk_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def inv_htk_mel(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def create_mel_fbanks(n_freqs=201, f_min=20.0, f_max=7600.0, n_mels=80, sample_rate=16000):
    m_min = htk_mel(f_min)
    m_max = htk_mel(f_max)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = inv_htk_mel(m_pts)
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    
    filter_diff = np.diff(f_pts)
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return torch.from_numpy(fb.astype(np.float32))

class Fbank(nn.Module):
    def __init__(self, n_mels=80, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, f_min=20.0, f_max=7600.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("fbanks", create_mel_fbanks(n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate))
        self.register_buffer("window", torch.hamming_window(win_length))

    @torch.no_grad()
    def forward(self, wav, lens):
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        power = stft.abs().pow(2)
        mel = torch.matmul(power.transpose(1, 2), self.fbanks)
        x = (mel + 1e-6).log()
        fl = 1 + lens // self.hop_length
        
        m = (torch.arange(x.size(1), device=lens.device)[None] < fl[:, None]).unsqueeze(-1)
        mu = (x * m).sum(1, keepdim=True) / fl[:, None, None]
        var = (((x - mu) ** 2) * m).sum(1, keepdim=True) / fl[:, None, None]
        return ((x - mu) / (var + 1e-5).sqrt()) * m, fl

# --- Conformer Architecture ---
class _FeedForwardModule(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, ffn, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, d, bias=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.sequential(x)

class _ConvolutionModule(nn.Module):
    def __init__(self, d, kernel=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.sequential = nn.Sequential(
            nn.Conv1d(d, 2 * d, 1, stride=1, padding=0, bias=True),
            nn.GLU(dim=1),
            nn.Conv1d(d, d, kernel, stride=1, padding=(kernel - 1) // 2, groups=d, bias=True),
            nn.BatchNorm1d(d),
            nn.SiLU(),
            nn.Conv1d(d, d, 1, stride=1, padding=0, bias=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)

class ConformerLayer(nn.Module):
    def __init__(self, d, heads, ffn, kernel, dropout=0.1):
        super().__init__()
        self.ffn1 = _FeedForwardModule(d, ffn, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.conv_module = _ConvolutionModule(d, kernel, dropout)
        self.ffn2 = _FeedForwardModule(d, ffn, dropout)
        self.final_layer_norm = nn.LayerNorm(d)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        sa_norm = self.self_attn_layer_norm(x)
        sa_out, _ = self.self_attn(sa_norm, sa_norm, sa_norm, key_padding_mask=key_padding_mask)
        x = x + sa_out
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_layer_norm(x)

class Conformer(nn.Module):
    def __init__(self, d, heads, ffn, layers, kernel, dropout=0.1):
        super().__init__()
        self.conformer_layers = nn.ModuleList([
            ConformerLayer(d, heads, ffn, kernel, dropout) for _ in range(layers)
        ])

    def forward(self, x, lengths):
        max_len = x.size(1)
        mask = torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]
        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask=mask)
        return x, lengths

def sinusoid(T, d, dev):
    pos = torch.arange(T, device=dev)[:, None]
    ang = pos / (10000 ** (torch.arange(0, d, 2, device=dev)[None] / d))
    pe = torch.zeros(T, d, device=dev)
    pe[:, 0::2], pe[:, 1::2] = ang.sin(), ang.cos()
    return pe

def len_mask(lens, T):
    return torch.arange(T, device=lens.device)[None] < lens[:, None]

def masked_mean(x, lens):
    m = len_mask(lens, x.size(1)).unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)

class Subsample(nn.Module):
    def __init__(self, d, c=32, n_mels=80):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, c, 3, 2), nn.ReLU(), nn.Conv2d(c, c, 3, 2), nn.ReLU())
        self.out = nn.Linear(c * (((n_mels - 3) // 2 + 1 - 3) // 2 + 1), d)
    def forward(self, x, lens):
        x = self.conv(x.unsqueeze(1))
        B, c, T, Fq = x.shape
        return self.out(x.transpose(1, 2).reshape(B, T, c * Fq)), (((lens - 3) // 2 + 1 - 3) // 2 + 1).clamp(min=1)

class ASP(nn.Module):
    def __init__(self, d, out=192, att=128):
        super().__init__()
        self.c1, self.c2, self.proj = nn.Conv1d(d, att, 1), nn.Conv1d(att, d, 1), nn.Linear(2 * d, out)
    def forward(self, x, lens):
        m = len_mask(lens, x.size(1))
        x = x.transpose(1, 2)
        a = self.c2(torch.tanh(self.c1(x))).masked_fill(~m[:, None], float("-inf")).softmax(-1)
        mu = (a * x).sum(-1)
        sd = ((a * x * x).sum(-1) - mu ** 2).clamp(min=1e-5).sqrt()
        return self.proj(torch.cat([mu, sd], -1))

class Student(nn.Module):
    def __init__(self, d=256, blocks=6, layers=2, heads=4, ffn=1024, kernel=15, sv_block=3, sv_dim=192, n_at=527, w_dim=1024):
        super().__init__()
        self.cfg = dict(d=d, blocks=blocks, layers=layers, heads=heads, ffn=ffn, kernel=kernel,
                        sv_block=sv_block, sv_dim=sv_dim, n_at=n_at, w_dim=w_dim)
        self.d, self.sv_block, self.sv_dim = d, sv_block, sv_dim
        self.sub = Subsample(d)
        self.blocks = nn.ModuleList([Conformer(d, heads, ffn, layers, kernel, dropout=0.1) for _ in range(blocks)])
        self.asr_proj = nn.Linear(d, w_dim)
        self.at_head = nn.Linear(d, n_at)
        self.sv_pool = ASP(d, sv_dim)

    def encode(self, fb, lens):
        x, lens = self.sub(fb, lens)
        x = x + sinusoid(x.size(1), self.d, x.device)
        mid = None
        for i, b in enumerate(self.blocks, 1):
            x, lens = b(x, lens)
            if i == self.sv_block:
                mid = x
        return x, mid, lens

    def at_logits(self, h, lens, head=None):
        return masked_mean((self.at_head if head is None else head)(h), lens)

    def sv_embed(self, mid, lens):
        return self.sv_pool(mid, lens)

    def add_heads(self, vocab, n_at, n_spk):
        self.ctc = nn.Linear(self.d, vocab)
        self.at41 = nn.Linear(self.d, n_at)
        self.spk_cls = nn.Linear(self.sv_dim, n_spk)

def ctc_collapse(seq, blank=0):
    res, prev = [], None
    for tok in seq:
        if tok != prev:
            if tok != blank:
                res.append(tok)
            prev = tok
    return res

# --- MT2KD Inference Engine ---
class MT2KDEngine:
    def __init__(self, model_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "model")
        self.device = device
        
        ck_path = os.path.join(model_dir, "mt2kd_model.pt")
        bpe_path = os.path.join(model_dir, "bpe.model")
        labels_path = os.path.join(model_dir, "labels.json")
        
        ck = torch.load(ck_path, map_location=device)
        self.model = Student(**ck["cfg"])
        self.sp = spm.SentencePieceProcessor(model_file=bpe_path)
        self.model.add_heads(self.sp.get_piece_size(), ck["heads"]["n_at"], ck["heads"]["n_spk"])
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.to(device)
        self.model.eval()
        
        self.fbank = Fbank().to(device)
        self.labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}

    def load_audio(self, source, target_sr=16000):
        # source can be file path, bytes, or file-like object
        if isinstance(source, (str, os.PathLike)):
            sr, data = wavfile.read(source)
        elif isinstance(source, bytes):
            sr, data = wavfile.read(io.BytesIO(source))
        else:
            sr, data = wavfile.read(source)

        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples).astype(np.float32)

        return torch.from_numpy(data)

    @torch.no_grad()
    def encode(self, wav_tensor):
        wav = wav_tensor.unsqueeze(0).to(self.device)
        lens = torch.tensor([wav.size(1)], device=self.device)
        fb, fl = self.fbank(wav, lens)
        h, mid, ol = self.model.encode(fb, fl)
        return h, mid, ol

    @torch.no_grad()
    def transcribe(self, wav_source):
        w = self.load_audio(wav_source)
        h, _, _ = self.encode(w)
        tokens = ctc_collapse(self.model.ctc(h).argmax(-1)[0].tolist())
        return self.sp.decode(tokens)

    @torch.no_grad()
    def audio_tag(self, wav_source, top_k=5):
        w = self.load_audio(wav_source)
        h, _, ol = self.encode(w)
        logits = self.model.at_logits(h, ol, self.model.at41)
        probs = torch.softmax(logits, dim=-1)[0]
        
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
        tags = []
        for prob, idx in zip(top_probs, top_indices):
            class_id = int(idx.item())
            class_name = self.labels.get(str(class_id), f"Class {class_id}")
            tags.append({
                "label": class_name,
                "score": round(float(prob.item()), 4)
            })
        return tags

    @torch.no_grad()
    def speaker_embedding(self, wav_source):
        w = self.load_audio(wav_source)
        _, mid, ol = self.encode(w)
        emb = F.normalize(self.model.sv_embed(mid, ol), dim=-1)[0]
        return emb.cpu().numpy().tolist()

    @torch.no_grad()
    def verify(self, wav1_source, wav2_source, threshold=0.50):
        emb1 = np.array(self.speaker_embedding(wav1_source))
        emb2 = np.array(self.speaker_embedding(wav2_source))
        score = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        return {
            "score": round(score, 4),
            "same_speaker": bool(score >= threshold),
            "threshold": threshold
        }

    @torch.no_grad()
    def analyze_all(self, wav_source):
        t0 = time.perf_counter()
        w = self.load_audio(wav_source)
        h, mid, ol = self.encode(w)
        
        # 1. ASR
        tokens = ctc_collapse(self.model.ctc(h).argmax(-1)[0].tolist())
        transcript = self.sp.decode(tokens)
        
        # 2. AT
        logits = self.model.at_logits(h, ol, self.model.at41)
        probs = torch.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, min(5, len(probs)))
        tags = []
        for prob, idx in zip(top_probs, top_indices):
            class_id = int(idx.item())
            tags.append({
                "label": self.labels.get(str(class_id), f"Class {class_id}"),
                "score": round(float(prob.item()), 4)
            })

        # 3. SV
        emb = F.normalize(self.model.sv_embed(mid, ol), dim=-1)[0].cpu().numpy().tolist()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "model": "MT2KD Student Conformer (19.0M parameters)",
            "latency_ms": latency_ms,
            "transcript": transcript,
            "tags": tags,
            "speaker": {"embedding": emb}
        }
