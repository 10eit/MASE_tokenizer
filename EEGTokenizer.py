import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class MASEConfig:
    fs: int = 256
    hidden_dim: int = 256
    morph_dim: int = 160
    local_spectral_dim: int = 32
    context_spectral_dim: int = 32
    state_dim: int = 32
    patch_size: int = 32
    stride: int = 16
    local_spectral_window: int = 128
    spectral_window: int = 256
    fmin: float = 0.5
    fmax: float = 40.0
    spectral_dropout: float = 0.25
    eps: float = 1e-6
    depth: int = 4
    heads: int = 8
    mlp_dim: int = 768
    dropout: float = 0.1
    proj_dim: int = 256
    attention_temperature: float = 1.0

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

class EEGTokenizer(nn.Module):

    def __init__(
        self,
        fs: int = 256,
        hidden_dim: int = 256,
        morph_dim: int = 160,
        local_spectral_dim: int = 32,
        context_spectral_dim: int = 32,
        state_dim: int = 32,
        patch_size: int = 32,
        stride: int = 16,
        local_spectral_window: int = 128,
        context_spectral_window: int = 256,
        fmin: float = 0.5,
        fmax: float = 40.0,
        spectral_dropout: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(local_spectral_window, context_spectral_window) < patch_size:
            raise ValueError("spectral windows must be >= patch_size")
        if patch_size < 2 or stride < 1:
            raise ValueError("patch_size must be >= 2 and stride must be positive")
        self.fs = fs
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stride = stride
        self.local_spectral_window = local_spectral_window
        self.context_spectral_window = context_spectral_window

        self.spectral_window = max(local_spectral_window, context_spectral_window)
        self.spectral_dropout = spectral_dropout
        self.eps = eps

        local_slice = self._frequency_slice(local_spectral_window, fs, fmin, fmax)
        context_slice = self._frequency_slice(context_spectral_window, fs, fmin, fmax)
        self.local_freq_start, self.local_freq_stop = local_slice
        self.context_freq_start, self.context_freq_stop = context_slice
        self.num_freq_bins = self.local_freq_stop - self.local_freq_start
        self.context_num_freq_bins = self.context_freq_stop - self.context_freq_start
        self.register_buffer(
            "local_hann_window",
            torch.hann_window(local_spectral_window),
            persistent=False,
        )
        self.register_buffer(
            "context_hann_window",
            torch.hann_window(context_spectral_window),
            persistent=False,
        )

        self.morphology = nn.Sequential(
            nn.LayerNorm(3 * patch_size),
            nn.Linear(3 * patch_size, morph_dim),
            nn.GELU(),
            nn.Linear(morph_dim, morph_dim),
        )
        self.local_spectral = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, local_spectral_dim),
            nn.GELU(),
            nn.Linear(local_spectral_dim, local_spectral_dim),
        )
        self.context_spectral = nn.Sequential(
            nn.LayerNorm(self.context_num_freq_bins),
            nn.Linear(self.context_num_freq_bins, context_spectral_dim),
            nn.GELU(),
            nn.Linear(context_spectral_dim, context_spectral_dim),
        )
        self.state = nn.Sequential(
            nn.Linear(3, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        fusion_dim = morph_dim + local_spectral_dim + context_spectral_dim + state_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _frequency_slice(
        window: int, fs: int, fmin: float, fmax: float
    ) -> tuple[int, int]:
        frequencies = torch.fft.rfftfreq(window, d=1.0 / fs)
        indices = (
            ((frequencies >= fmin) & (frequencies <= fmax))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if indices.numel() == 0:
            raise ValueError("No FFT bins fall inside [fmin, fmax]")
        return int(indices[0]), int(indices[-1]) + 1

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.size(1) != 1:
                raise ValueError(f"Expected single-channel EEG, got {tuple(x.shape)}")
            x = x[:, 0]
        if x.ndim != 2:
            raise ValueError(f"Expected [B,T] or [B,1,T], got {tuple(x.shape)}")
        if x.size(-1) < self.patch_size:
            raise ValueError("Input shorter than patch_size")
        return x

    def _center(
        self, x: torch.Tensor, region_start: int = 0, region_length: int | None = None
    ) -> torch.Tensor:
        region = x[:, region_start : region_start + region_length]
        return x - region.median(dim=-1, keepdim=True).values

    def _unit_rms(
        self, x: torch.Tensor, region_start: int = 0, region_length: int | None = None
    ) -> torch.Tensor:
        region = x[:, region_start : region_start + region_length]
        rms = region.square().mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps)

    @staticmethod
    def _diff(x: torch.Tensor) -> torch.Tensor:
        derivative = x[:, 1:] - x[:, :-1]
        return torch.cat((derivative, derivative[:, -1:]), dim=-1)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        return x.unfold(-1, self.patch_size, self.stride)

    @staticmethod
    def _as_shared_int(value: int | torch.Tensor | None, name: str) -> int | None:
        if value is None:
            return None
        if torch.is_tensor(value):
            flattened = value.detach().cpu().reshape(-1)
            if flattened.numel() == 0 or not torch.all(flattened == flattened[0]):
                raise ValueError(f"All examples in a batch must share {name}")
            return int(flattened[0])
        return int(value)

    def normalization_region(
        self,
        total_samples: int,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
    ) -> tuple[int, int]:
        core_start = self._as_shared_int(core_start, "core_start")
        core_length = self._as_shared_int(core_length, "core_length")
        if core_start is None and core_length is None:
            return 0, total_samples
        if core_start is None or core_length is None:
            raise ValueError("core_start and core_length must be supplied together")
        if (
            core_start < 0
            or core_length < 1
            or core_start + core_length > total_samples
        ):
            raise ValueError("Core is outside the supplied EEG slice")
        return core_start, core_length

    def _spectral_contexts(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        total_pad = window_size - self.patch_size
        left = total_pad // 2
        right = total_pad - left
        mode = "reflect" if x.size(-1) > max(left, right) else "replicate"
        padded = F.pad(x[:, None], (left, right), mode=mode)[:, 0]
        return padded.unfold(-1, window_size, self.stride)

    def _spectral_distribution(
        self,
        x: torch.Tensor,
        window_size: int,
        hann_window: torch.Tensor,
        freq_start: int,
        freq_stop: int,
    ) -> torch.Tensor:
        contexts = self._spectral_contexts(x, window_size)
        contexts = contexts - contexts.mean(dim=-1, keepdim=True)
        window = hann_window.to(dtype=contexts.dtype)

        with torch.autocast(device_type=x.device.type, enabled=False):
            fft = torch.fft.rfft(contexts.float() * window.float(), dim=-1)
            power = fft.real.square() + fft.imag.square()
            power = power[..., freq_start:freq_stop]
            distribution = power / (power.sum(dim=-1, keepdim=True) + self.eps)
        return distribution.to(dtype=x.dtype)

    def _state_features(self, centered: torch.Tensor) -> torch.Tensor:
        raw_patches = self._patches(centered)
        rms = raw_patches.square().mean(dim=-1).add(self.eps).sqrt()
        peak_to_peak = raw_patches.amax(dim=-1) - raw_patches.amin(dim=-1)
        line_length = (raw_patches[..., 1:] - raw_patches[..., :-1]).abs().mean(dim=-1)
        return torch.stack(
            (
                torch.log(rms + self.eps),
                torch.log(peak_to_peak + self.eps),
                torch.log(line_length / (rms + self.eps) + self.eps),
            ),
            dim=-1,
        )

    def token_slice(
        self,
        total_samples: int,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
    ) -> slice:
        core_start = self._as_shared_int(core_start, "core_start")
        core_length = self._as_shared_int(core_length, "core_length")
        token_count = (total_samples - self.patch_size) // self.stride + 1
        if core_start is None and core_length is None:
            return slice(0, token_count)
        if core_start is None or core_length is None:
            raise ValueError("core_start and core_length must be supplied together")
        if core_start < 0 or core_length < self.patch_size:
            raise ValueError("Invalid core bounds")
        first = math.ceil(core_start / self.stride)
        last = math.floor((core_start + core_length - self.patch_size) / self.stride)
        if first > last or last >= token_count:
            raise ValueError("Core is outside the supplied EEG slice")
        total_pad = self.spectral_window - self.patch_size
        spectral_left = total_pad // 2
        spectral_right = total_pad - spectral_left
        if (
            first * self.stride < spectral_left
            or last * self.stride + self.patch_size + spectral_right > total_samples
        ):
            raise ValueError(
                "The retained core tokens do not have enough real spectral context; "
                "read more continuous EEG on both sides"
            )
        return slice(first, last + 1)

    @property
    def required_context(self) -> tuple[int, int]:
        total_pad = self.spectral_window - self.patch_size
        return total_pad // 2, total_pad - total_pad // 2

    def _features(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None,
        core_length: int | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prepared = self._prepare(x)
        region_start, region_length = self.normalization_region(
            prepared.size(-1), core_start, core_length
        )
        centered = self._center(prepared, region_start, region_length)
        shape = self._unit_rms(centered, region_start, region_length)

        first_derivative = self._diff(shape)
        second_derivative = self._diff(first_derivative)
        morphology = torch.cat(
            (
                self._patches(shape),
                self._patches(first_derivative),
                self._patches(second_derivative),
            ),
            dim=-1,
        )
        local_spectrum = self._spectral_distribution(
            shape,
            self.local_spectral_window,
            self.local_hann_window,
            self.local_freq_start,
            self.local_freq_stop,
        )
        context_spectrum = self._spectral_distribution(
            shape,
            self.context_spectral_window,
            self.context_hann_window,
            self.context_freq_start,
            self.context_freq_stop,
        )
        return (
            morphology,
            local_spectrum,
            context_spectrum,
            self._state_features(centered),
            shape,
        )

    def make_targets(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        morphology, local_spectrum, _context_spectrum, state, shape = self._features(
            x, core_start, core_length
        )
        del morphology
        selection = self.token_slice(shape.size(-1), core_start, core_length)
        return {
            "wave": self._patches(shape)[:, selection].detach(),
            "spectrum": local_spectrum[:, selection].detach(),
            "state": state[:, selection].detach(),
        }

    def forward(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        prepared = self._prepare(x)
        morphology, local_spectrum, context_spectrum, state, shape = self._features(
            prepared, core_start, core_length
        )
        selection = self.token_slice(prepared.size(-1), core_start, core_length)
        morphology = morphology[:, selection]
        local_spectrum = local_spectrum[:, selection]
        context_spectrum = context_spectrum[:, selection]
        state = state[:, selection]
        morphology_embedding = self.morphology(morphology)
        local_embedding = self.local_spectral(torch.log(local_spectrum + self.eps))
        context_embedding = self.context_spectral(
            torch.log(context_spectrum + self.eps)
        )
        spectral_drop_mask = None
        if self.training and self.spectral_dropout > 0:
            spectral_drop_mask = (
                torch.rand(local_embedding.shape[:2], device=x.device)
                < self.spectral_dropout
            )
            local_embedding = local_embedding.masked_fill(
                spectral_drop_mask[..., None], 0.0
            )
            context_embedding = context_embedding.masked_fill(
                spectral_drop_mask[..., None], 0.0
            )
        state_embedding = self.state(state)
        tokens = self.fusion(
            torch.cat(
                (
                    morphology_embedding,
                    local_embedding,
                    context_embedding,
                    state_embedding,
                ),
                dim=-1,
            )
        )
        if return_parts:
            return tokens, {
                "morphology": morphology_embedding,
                "spectral_local": local_embedding,
                "spectral_context": context_embedding,
                "state": state_embedding,
                "spectral_drop_mask": spectral_drop_mask,

                "wave_target": self._patches(shape)[:, selection].detach(),
                "spectrum_target": local_spectrum.detach(),
            }
        return tokens

def sinusoidal_position_encoding(
    length: int, dimension: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    indices = torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
    divisor = torch.exp(-math.log(10000.0) * indices / dimension)
    angles = position * divisor
    encoding = torch.empty(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if dimension > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : dimension // 2])
    return encoding.to(dtype=dtype)

class EEGTransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, hidden_dim = x.shape
        qkv = self.qkv(self.attention_norm(x)).reshape(
            batch_size, token_count, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention = (
            attention.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, token_count, hidden_dim)
        )
        x = x + self.attention_dropout(self.attention_output(attention))
        return x + self.mlp(self.mlp_norm(x))

class EEGSegmentEncoder(nn.Module):
    def __init__(self, config: MASEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MASEConfig()
        config = self.config
        if config.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        self.tokenizer = EEGTokenizer(
            fs=config.fs,
            hidden_dim=config.hidden_dim,
            morph_dim=config.morph_dim,
            local_spectral_dim=config.local_spectral_dim,
            context_spectral_dim=config.context_spectral_dim,
            state_dim=config.state_dim,
            patch_size=config.patch_size,
            stride=config.stride,
            local_spectral_window=config.local_spectral_window,
            context_spectral_window=config.spectral_window,
            fmin=config.fmin,
            fmax=config.fmax,
            spectral_dropout=config.spectral_dropout,
            eps=config.eps,
        )
        self.encoder = nn.ModuleList(
            EEGTransformerBlock(
                hidden_dim=config.hidden_dim,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.attention_pool = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attention_query = nn.Parameter(torch.empty(config.hidden_dim))
        nn.init.normal_(self.attention_query, std=config.hidden_dim**-0.5)
        self.local_head = nn.Sequential(
            nn.LayerNorm(2 * config.hidden_dim),
            nn.Linear(2 * config.hidden_dim, config.proj_dim),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.proj_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(config.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.wave_head = nn.Linear(config.hidden_dim, config.patch_size)
        self.spec_head = nn.Linear(config.hidden_dim, self.tokenizer.num_freq_bins)
        self.register_buffer("_position_cache", torch.empty(0), persistent=False)

    def _position_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        length, dimension = tokens.shape[1:]
        cache = self._position_cache
        if (
            cache.ndim != 2
            or cache.size(0) < length
            or cache.size(1) != dimension
            or cache.device != tokens.device
            or cache.dtype != tokens.dtype
        ):
            cache = sinusoidal_position_encoding(
                length, dimension, tokens.device, tokens.dtype
            )
            self._position_cache = cache
        return cache[:length]

    def forward(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
        *,
        token_mask: torch.Tensor | None = None,
        return_layer_means: bool = False,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        tokens, parts = self.tokenizer(x, core_start, core_length, return_parts=True)
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"token_mask has shape {tuple(token_mask.shape)}, expected "
                    f"{tuple(tokens.shape[:2])}"
                )
            tokens = torch.where(token_mask[..., None], self.mask_token, tokens)
        encoded = tokens + self._position_encoding(tokens)[None]
        layer_means = []
        hidden_states = []
        for block in self.encoder:
            encoded = block(encoded)
            if return_layer_means:
                layer_means.append(encoded.mean(dim=1))
            if return_hidden_states:
                hidden_states.append(encoded)
        encoded = self.norm(encoded)
        mean = encoded.mean(dim=1)
        attention_logits = (
            torch.tanh(self.attention_pool(encoded))
            * self.attention_query.to(dtype=encoded.dtype)
        ).sum(dim=-1)
        attention = (attention_logits / self.config.attention_temperature).softmax(
            dim=-1
        )
        attended = torch.sum(encoded * attention[..., None], dim=1)
        pooled = torch.cat((mean, attended), dim=-1)
        return {
            "tokens": encoded,
            "layer_means": (
                torch.stack(layer_means, dim=1) if return_layer_means else None
            ),
            "hidden_states": (
                torch.stack(hidden_states, dim=1) if return_hidden_states else None
            ),
            "local": self.local_head(pooled),
            "global": self.global_head(mean),
            "wave": self.wave_head(encoded),
            "spectrum_logits": self.spec_head(encoded),
            "spectral_drop_mask": parts["spectral_drop_mask"],
            "wave_target": parts.get("wave_target"),
            "spectrum_target": parts.get("spectrum_target"),
            "token_mask": token_mask,
        }

    @torch.no_grad()
    def encode_layer_mean_std(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
        *,
        layer: int,
    ) -> torch.Tensor:
        if self.training:
            raise ValueError("Layer readout requires model.eval()")
        if not 1 <= layer <= len(self.encoder):
            raise ValueError(f"Layer {layer} is outside this encoder")
        tokens = self.tokenizer(x, core_start, core_length)
        encoded = tokens + self._position_encoding(tokens)[None]
        for block in self.encoder[:layer]:
            encoded = block(encoded)
        return F.normalize(
            torch.cat((encoded.mean(dim=1), encoded.std(dim=1, correction=0)), dim=-1),
            dim=-1,
        )

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        core_start: int | torch.Tensor | None = None,
        core_length: int | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.forward(x, core_start, core_length)
        return {
            "tokens": F.normalize(output["tokens"], dim=-1),
            "local": F.normalize(output["local"], dim=-1),
            "global": F.normalize(output["global"], dim=-1),
        }

@dataclass(frozen=True)
class MASETrainerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    temperature: float = 0.1
    local_weight: float = 1.0
    token_weight: float = 0.5
    global_weight: float = 0.25
    wave_weight: float = 0.2
    spectrum_weight: float = 0.1
    derivative_weight: float = 0.25
    cross_duration_weight: float = 1.0
    vicreg_invariance_weight: float = 25.0
    vicreg_variance_weight: float = 25.0
    vicreg_covariance_weight: float = 1.0
    mask_ratio: float = 0.35
    mask_span_min: int = 2
    mask_span_max: int = 4
    max_token_matches: int = 8
    noise_std: float = 0.02
    gain_jitter: float = 0.05
    sign_flip_probability: float = 0.10
    grad_clip: float = 1.0

class MASETrainer:

    def __init__(
        self,
        model: EEGSegmentEncoder,
        config: MASETrainerConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config or MASETrainerConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        dimensions = tuple(range(1, x.ndim))
        scale = x.std(dim=dimensions, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x) * scale * self.config.noise_std
        gain_shape = (x.size(0),) + (1,) * (x.ndim - 1)
        gain = 1.0 + torch.randn(
            gain_shape, device=x.device, dtype=x.dtype
        ) * self.config.gain_jitter
        sign = torch.where(
            torch.rand(gain_shape, device=x.device)
            < self.config.sign_flip_probability,
            -torch.ones(gain_shape, device=x.device, dtype=x.dtype),
            torch.ones(gain_shape, device=x.device, dtype=x.dtype),
        )
        return x * gain.clamp(0.8, 1.2) * sign + noise

    def _mask(
        self, batch_size: int, token_count: int, deterministic: bool
    ) -> torch.Tensor:
        target = max(1, int(round(token_count * self.config.mask_ratio)))
        span_max = min(self.config.mask_span_max, token_count)
        span_min = min(self.config.mask_span_min, span_max)
        span_mean = (span_min + span_max) / 2
        span_count = max(1, math.ceil(target / span_mean))
        start_options = token_count - span_max + 1
        if deterministic:
            starts = (
                torch.arange(batch_size, device=self.device)[:, None]
                + torch.arange(span_count, device=self.device)[None] * 5
            ) % start_options
            lengths = torch.full(
                (batch_size, span_count),
                span_min,
                device=self.device,
                dtype=torch.long,
            )
        else:
            starts = torch.randint(
                start_options, (batch_size, span_count), device=self.device
            )
            lengths = torch.randint(
                span_min, span_max + 1, (batch_size, span_count), device=self.device
            )
        offsets = torch.arange(span_max, device=self.device)
        positions = starts[..., None] + offsets
        active = offsets[None, None] < lengths[..., None]
        mask = torch.zeros(
            batch_size, token_count, dtype=torch.bool, device=self.device
        )
        mask.scatter_(1, positions.flatten(1), active.flatten(1))
        return mask

    def _vicreg(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = first.float()
            second = second.float()
            invariance = F.mse_loss(first, second)
            if first.size(0) < 2:
                return self.config.vicreg_invariance_weight * invariance
            first_centered = first - first.mean(dim=0)
            second_centered = second - second.mean(dim=0)
            first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
            second_std = torch.sqrt(
                second_centered.var(dim=0, unbiased=False) + 1e-4
            )
            variance = 0.5 * (
                F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
            )
            denominator = first.size(0) - 1
            first_cov = first_centered.T @ first_centered / denominator
            second_cov = second_centered.T @ second_centered / denominator
            dimension = first.size(1)
            identity = torch.eye(dimension, device=first.device, dtype=torch.bool)
            covariance = 0.5 * (
                first_cov.masked_select(~identity).square().sum() / dimension
                + second_cov.masked_select(~identity).square().sum() / dimension
            )
            return (
                self.config.vicreg_invariance_weight * invariance
                + self.config.vicreg_variance_weight * variance
                + self.config.vicreg_covariance_weight * covariance
            )

    def _info_nce(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.size(0) < 2:
            return first.sum() * 0.0
        with torch.autocast(device_type=first.device.type, enabled=False):
            first = F.normalize(first.float(), dim=-1)
            second = F.normalize(second.float(), dim=-1)
            logits = (first @ second.T) / self.config.temperature
            labels = torch.arange(first.size(0), device=first.device)
            return 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
            )

    def _matched_tokens(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        length_a: int,
        length_b: int,
        shift_a: torch.Tensor,
        shift_b: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.model.config.patch_size
        stride = self.model.config.stride
        token_a = torch.arange(first.size(1), device=first.device)
        starts_a = shift_a[:, None] - length_a // 2 + token_a[None] * stride
        first_center = starts_a + patch / 2
        second_origin = shift_b[:, None] - length_b // 2 + patch / 2
        nearest_b = torch.round((first_center - second_origin) / stride).long()
        nearest_b_clamped = nearest_b.clamp(0, second.size(1) - 1)
        second_center = second_origin + nearest_b_clamped * stride
        overlap = patch - torch.abs(first_center - second_center)
        valid = (nearest_b >= 0) & (nearest_b < second.size(1)) & (overlap > 0)
        minimum = int(valid.sum(dim=1).min())
        if minimum == 0:
            zero = first.reshape(-1, first.size(-1))[:0]
            return zero, zero
        count = min(self.config.max_token_matches, minimum)
        if deterministic:
            scores = -torch.arange(first.size(1), device=first.device).float()
            scores = scores[None].expand(first.size(0), -1)
        else:
            scores = torch.rand(valid.shape, device=first.device)
        indices_a = scores.masked_fill(~valid, float("-inf")).topk(
            count, dim=1
        ).indices
        indices_b = torch.gather(nearest_b_clamped, 1, indices_a)
        feature_index_a = indices_a[..., None].expand(-1, -1, first.size(-1))
        feature_index_b = indices_b[..., None].expand(-1, -1, second.size(-1))
        return (
            torch.gather(first, 1, feature_index_a).flatten(0, 1),
            torch.gather(second, 1, feature_index_b).flatten(0, 1),
        )

    def _wave_loss(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        prediction = prediction[mask]
        target = target[mask]
        waveform = F.smooth_l1_loss(prediction, target)
        prediction_diff = prediction[..., 1:] - prediction[..., :-1]
        target_diff = target[..., 1:] - target[..., :-1]
        derivative = F.smooth_l1_loss(prediction_diff, target_diff)
        return waveform + self.config.derivative_weight * derivative

    @staticmethod
    def _spectrum_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        distribution = logits.softmax(dim=-1)
        hellinger = 1.0 - torch.sqrt((distribution * target).clamp_min(1e-12)).sum(
            dim=-1
        )
        return hellinger[mask].mean()

    def crop_view(
        self,
        read: torch.Tensor,
        anchor: torch.Tensor,
        length: int,
        shift: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        left, right = self.model.tokenizer.required_context
        starts = anchor + shift - length // 2 - left
        indices = starts[:, None] + torch.arange(
            length + left + right, device=read.device
        )[None]
        if int(indices.min()) < 0 or int(indices.max()) >= read.size(-1):
            raise ValueError("View falls outside the sampler's continuous read")
        return torch.gather(read[:, 0], 1, indices).unsqueeze(1), left

    def loss(
        self,
        read: torch.Tensor,
        anchor: torch.Tensor,
        *,
        length_a: int,
        length_b: int,
        shift_a: torch.Tensor,
        shift_b: torch.Tensor,
        cross_duration: bool,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        first_x, core_start = self.crop_view(read, anchor, length_a, shift_a)
        second_x, _ = self.crop_view(read, anchor, length_b, shift_b)
        if train:
            first_x = self._augment(first_x)
            second_x = self._augment(second_x)
        token_count_a = (length_a - self.model.config.patch_size) // self.model.config.stride + 1
        token_count_b = (length_b - self.model.config.patch_size) // self.model.config.stride + 1
        mask_a = self._mask(first_x.size(0), token_count_a, deterministic=not train)
        mask_b = (
            mask_a.clone()
            if token_count_a == token_count_b
            else self._mask(first_x.size(0), token_count_b, deterministic=not train)
        )
        if token_count_a == token_count_b and first_x.size(-1) == second_x.size(-1):
            combined = self.model(
                torch.cat((first_x, second_x), dim=0),
                core_start,
                length_a,
                token_mask=torch.cat((mask_a, mask_b), dim=0),
            )
            batch_size = first_x.size(0)
            first_output = {
                key: value[:batch_size] if torch.is_tensor(value) else value
                for key, value in combined.items()
            }
            second_output = {
                key: value[batch_size:] if torch.is_tensor(value) else value
                for key, value in combined.items()
            }
        else:
            first_output = self.model(
                first_x, core_start, length_a, token_mask=mask_a
            )
            second_output = self.model(
                second_x, core_start, length_b, token_mask=mask_b
            )
        local = self._vicreg(first_output["local"], second_output["local"])
        matched_a, matched_b = self._matched_tokens(
            first_output["tokens"],
            second_output["tokens"],
            length_a,
            length_b,
            shift_a,
            shift_b,
            deterministic=not train,
        )
        token = (
            self._vicreg(matched_a, matched_b)
            if matched_a.size(0)
            else first_output["tokens"].sum() * 0.0
        )
        if cross_duration:
            global_loss = (
                first_output["global"].sum() + second_output["global"].sum()
            ) * 0.0
        else:
            global_loss = self._info_nce(
                first_output["global"], second_output["global"]
            )
        wave = 0.5 * (
            self._wave_loss(first_output["wave"], first_output["wave_target"], mask_a)
            + self._wave_loss(
                second_output["wave"], second_output["wave_target"], mask_b
            )
        )
        spectrum = 0.5 * (
            self._spectrum_loss(
                first_output["spectrum_logits"],
                first_output["spectrum_target"],
                mask_a,
            )
            + self._spectrum_loss(
                second_output["spectrum_logits"],
                second_output["spectrum_target"],
                mask_b,
            )
        )
        alignment_weight = self.config.cross_duration_weight if cross_duration else 1.0
        total = (
            self.config.local_weight * alignment_weight * local
            + self.config.token_weight * alignment_weight * token
            + self.config.global_weight * global_loss
            + self.config.wave_weight * wave
            + self.config.spectrum_weight * spectrum
        )
        return {
            "loss": total,
            "local": alignment_weight * local.detach(),
            "token": alignment_weight * token.detach(),
            "global": global_loss.detach(),
            "wave": wave.detach(),
            "spectrum": spectrum.detach(),
        }

    def step(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        losses = self.loss(*args, **kwargs)
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return losses
