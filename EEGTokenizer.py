import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGTokenizer(nn.Module):
    def __init__(
        self,
        fs=256,
        hidden_dim=256,
        morph_dim=128,
        spectral_dim=96,
        state_dim=32,
        patch_size=32,
        stride=16,
        spectral_window=256,
        fmin=0.5,
        fmax=40.0,
        spectral_dropout=0.25,
        eps=1e-6,
    ):
        super().__init__()
        if spectral_window < patch_size:
            raise ValueError("spectral_window must be >= patch_size")

        self.fs = fs
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.stride = stride
        self.spectral_window = spectral_window
        self.spectral_dropout = spectral_dropout
        self.eps = eps

        freqs = torch.fft.rfftfreq(spectral_window, d=1.0 / fs)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freq_indices = freq_mask.nonzero(as_tuple=False).flatten()
        if freq_indices.numel() == 0:
            raise ValueError("No FFT bins fall inside [fmin, fmax]")

        self.freq_start = int(freq_indices[0])
        self.freq_stop = int(freq_indices[-1]) + 1
        self.num_freq_bins = self.freq_stop - self.freq_start
        self.register_buffer("freq_mask", freq_mask, persistent=False)
        self.register_buffer(
            "hann_window", torch.hann_window(spectral_window), persistent=False
        )

        self.morphology = nn.Sequential(
            nn.LayerNorm(3 * patch_size),
            nn.Linear(3 * patch_size, morph_dim),
            nn.GELU(),
            nn.Linear(morph_dim, morph_dim),
        )

        self.spectral = nn.Sequential(
            nn.LayerNorm(self.num_freq_bins),
            nn.Linear(self.num_freq_bins, spectral_dim),
            nn.GELU(),
            nn.Linear(spectral_dim, spectral_dim),
        )

        self.state = nn.Sequential(
            nn.Linear(3, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )

        fusion_dim = morph_dim + spectral_dim + state_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _prepare(self, x):
        if x.ndim == 3:
            if x.size(1) != 1:
                raise ValueError(f"Expected single-channel EEG, got {tuple(x.shape)}")
            x = x[:, 0]

        if x.ndim != 2:
            raise ValueError(f"Expected [B,T] or [B,1,T], got {tuple(x.shape)}")

        if x.size(-1) < self.patch_size:
            raise ValueError("Input shorter than patch_size")

        return x

    def _center(self, x):
        return x - x.median(dim=-1, keepdim=True).values

    def _unit_rms(self, x):
        rms = x.square().mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps)

    def _diff(self, x):
        d = x[:, 1:] - x[:, :-1]
        return torch.cat((d, d[:, -1:]), dim=-1)

    def _patches(self, x):
        return x.unfold(-1, self.patch_size, self.stride)

    def _spectral_contexts(self, x):
        total_pad = self.spectral_window - self.patch_size
        left = total_pad // 2
        right = total_pad - left
        mode = "reflect" if x.size(-1) > max(left, right) else "replicate"
        x = F.pad(x[:, None], (left, right), mode=mode)[:, 0]
        return x.unfold(-1, self.spectral_window, self.stride)

    def _spectral_distribution(self, x):
        ctx = self._spectral_contexts(x)
        ctx = ctx - ctx.mean(dim=-1, keepdim=True)
        window = self.hann_window.to(dtype=ctx.dtype)
        fft = torch.fft.rfft(ctx * window, dim=-1)


        power = fft.real.square() + fft.imag.square()
        power = power[..., self.freq_start : self.freq_stop]
        return power / (power.sum(dim=-1, keepdim=True) + self.eps)

    def _state_features(self, centered):
        rawp = self._patches(centered)
        rms = rawp.square().mean(dim=-1).add(self.eps).sqrt()
        p2p = rawp.amax(dim=-1) - rawp.amin(dim=-1)
        line_length = (rawp[..., 1:] - rawp[..., :-1]).abs().mean(dim=-1)
        return torch.stack(
            (
                torch.log(rms + self.eps),
                torch.log(p2p + self.eps),
                torch.log(line_length / (rms + self.eps) + self.eps),
            ),
            dim=-1,
        )

    def _features(self, x):
        centered = self._center(self._prepare(x))
        x_shape = self._unit_rms(centered)
        dx = self._unit_rms(self._diff(x_shape))
        d2x = self._unit_rms(self._diff(dx))
        wave = self._patches(x_shape)
        morph = torch.cat((wave, self._patches(dx), self._patches(d2x)), dim=-1)
        spectrum = self._spectral_distribution(x_shape)
        state = self._state_features(centered)
        return morph, torch.log(spectrum + self.eps), state, wave, spectrum

    def make_targets(self, x):
        centered = self._center(self._prepare(x))
        x_shape = self._unit_rms(centered)
        return {
            "wave": self._patches(x_shape).detach(),
            "spectrum": self._spectral_distribution(x_shape).detach(),
            "state": self._state_features(centered).detach(),
        }

    def forward(self, x, return_parts=False):
        morph, spec, state, _, _ = self._features(x)

        morph = self.morphology(morph)
        spec = self.spectral(spec)
        state = self.state(state)

        spectral_drop_mask = None
        if self.training and self.spectral_dropout > 0:
            spectral_drop_mask = torch.rand(spec.shape[:2], device=spec.device) < (
                self.spectral_dropout
            )
            spec = spec.masked_fill(spectral_drop_mask[..., None], 0.0)

        tokens = self.fusion(torch.cat((morph, spec, state), dim=-1))

        if return_parts:
            return tokens, {
                "morphology": morph,
                "spectral": spec,
                "state": state,
                "spectral_drop_mask": spectral_drop_mask,
            }

        return tokens


def sinusoidal_position_encoding(n, d, device, dtype):
    position = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    idx = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * idx / d)
    angles = position * div
    pe = torch.empty(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    if d > 1:
        pe[:, 1::2] = torch.cos(angles[:, : d // 2])
    return pe.to(dtype=dtype)


class EEGSegmentEncoder(nn.Module):
    def __init__(
        self,
        tokenizer=None,
        hidden_dim=256,
        depth=4,
        heads=8,
        mlp_dim=768,
        dropout=0.1,
        proj_dim=256,
    ):
        super().__init__()

        self.tokenizer = tokenizer or EEGTokenizer(hidden_dim=hidden_dim)

        if self.tokenizer.hidden_dim != hidden_dim:
            raise ValueError("tokenizer.hidden_dim != hidden_dim")

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_dim)
        self.segment_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.wave_head = nn.Linear(hidden_dim, self.tokenizer.patch_size)
        self.spec_head = nn.Linear(hidden_dim, self.tokenizer.num_freq_bins)
        self.register_buffer("_position_cache", torch.empty(0), persistent=False)

    def _position_encoding(self, z):
        n, d = z.shape[1:]
        cache = self._position_cache
        needs_refresh = (
            cache.ndim != 2
            or cache.size(0) < n
            or cache.size(1) != d
            or cache.device != z.device
            or cache.dtype != z.dtype
        )
        if needs_refresh:
            cache = sinusoidal_position_encoding(n, d, z.device, z.dtype)
            self._position_cache = cache
        return cache[:n]

    def forward(self, x):
        z, parts = self.tokenizer(x, return_parts=True)
        z = self.norm(self.encoder(z + self._position_encoding(z)[None]))
        segment = self.segment_head(z.mean(dim=1))
        return {
            "tokens": z,
            "segment": segment,
            "wave": self.wave_head(z),
            "spectrum_logits": self.spec_head(z),
            "spectral_drop_mask": parts["spectral_drop_mask"],
        }

    @torch.no_grad()
    def encode(self, x):
        out = self.forward(x)
        return {
            "tokens": F.normalize(out["tokens"], dim=-1),
            "segment": F.normalize(out["segment"], dim=-1),
        }


class Trainer:
    def __init__(
        self,
        model,
        lr=3e-4,
        weight_decay=1e-2,
        temperature=0.1,
        token_weight=1.0,
        segment_weight=0.5,
        wave_weight=0.2,
        spectrum_weight=0.2,
        derivative_weight=0.25,
        noise_std=0.02,
        gain_jitter=0.05,
        grad_clip=1.0,
        grad_accum_steps=1,
        amp=True,
        device=None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.temperature = temperature
        self.token_weight, self.segment_weight = token_weight, segment_weight
        self.wave_weight, self.spectrum_weight = wave_weight, spectrum_weight
        self.derivative_weight = derivative_weight
        self.noise_std, self.gain_jitter = noise_std, gain_jitter
        self.grad_clip, self.grad_accum_steps = grad_clip, grad_accum_steps
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.use_amp = amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if (self.use_amp and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )
        self.scheduler = None

    def _get_x(self, batch):
        if torch.is_tensor(batch):
            x = batch
        elif isinstance(batch, dict):
            x = batch["x"]
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return x.to(device=self.device, dtype=torch.float32, non_blocking=True)

    def _augment(self, x, scale=None):
        dims = tuple(range(1, x.ndim))
        if scale is None:
            scale = x.std(dim=dims, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x) * scale * self.noise_std
        gain_shape = (x.size(0),) + (1,) * (x.ndim - 1)
        gain = (
            1.0
            + torch.randn(gain_shape, device=x.device, dtype=x.dtype) * self.gain_jitter
        )
        return x * gain.clamp(0.8, 1.2) + noise

    def _augment_pair(self, x):
        scale = x.std(dim=tuple(range(1, x.ndim)), keepdim=True).clamp_min(1e-6)
        return self._augment(x, scale), self._augment(x, scale)

    def _info_nce(self, a, b):
        if a.size(0) < 2:
            return a.new_zeros(())

        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = (a @ b.T) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

    def _token_info_nce(self, a, b):
        if a.size(0) < 2:
            return a.new_zeros(())

        a = F.normalize(a, dim=-1).transpose(0, 1)
        b = F.normalize(b, dim=-1).transpose(0, 1)
        logits = torch.bmm(a, b.transpose(1, 2)) / self.temperature
        n, bs, _ = logits.shape
        labels = torch.arange(bs, device=a.device).repeat(n)
        forward_loss = F.cross_entropy(logits.reshape(n * bs, bs), labels)
        backward_loss = F.cross_entropy(
            logits.transpose(1, 2).reshape(n * bs, bs),
            labels,
        )
        return 0.5 * (forward_loss + backward_loss)

    def _wave_loss(self, pred, target):
        waveform = F.smooth_l1_loss(pred, target)
        pred_diff = pred[..., 1:] - pred[..., :-1]
        target_diff = target[..., 1:] - target[..., :-1]
        derivative = F.smooth_l1_loss(pred_diff, target_diff)
        return waveform + self.derivative_weight * derivative

    def _spectrum_loss(self, logits, target, drop_mask=None):
        q = logits.softmax(dim=-1)
        hellinger = 1.0 - torch.sqrt((q * target).clamp_min(1e-12)).sum(dim=-1)
        if drop_mask is None:
            return hellinger.mean()

        weights = drop_mask.to(dtype=hellinger.dtype)
        count = weights.sum()
        masked_mean = (hellinger * weights).sum() / count.clamp_min(1.0)
        return torch.where(count > 0, masked_mean, hellinger.mean())

    def _loss(self, x, augment=True):
        with torch.no_grad():
            target = self.model.tokenizer.make_targets(x)

        if augment:
            v1, v2 = self._augment_pair(x)
            combined = self.model(torch.cat((v1, v2), dim=0))
            batch_size = x.size(0)
            out1 = {
                key: value[:batch_size] if value is not None else None
                for key, value in combined.items()
            }
            out2 = {
                key: value[batch_size:] if value is not None else None
                for key, value in combined.items()
            }
        else:
            out1 = out2 = self.model(x)

        token_loss = self._token_info_nce(out1["tokens"], out2["tokens"])
        segment_loss = self._info_nce(out1["segment"], out2["segment"])
        wave_loss = self._wave_loss(out1["wave"], target["wave"])
        spectrum_loss = self._spectrum_loss(
            out1["spectrum_logits"],
            target["spectrum"],
            out1["spectral_drop_mask"],
        )
        if augment:
            wave_loss = 0.5 * (
                wave_loss + self._wave_loss(out2["wave"], target["wave"])
            )
            spectrum_loss = 0.5 * (
                spectrum_loss
                + self._spectrum_loss(
                    out2["spectrum_logits"],
                    target["spectrum"],
                    out2["spectral_drop_mask"],
                )
            )

        loss = (
            self.token_weight * token_loss
            + self.segment_weight * segment_loss
            + self.wave_weight * wave_loss
            + self.spectrum_weight * spectrum_loss
        )
        return {
            "loss": loss,
            "token": token_loss.detach(),
            "segment": segment_loss.detach(),
            "wave": wave_loss.detach(),
            "spectrum": spectrum_loss.detach(),
        }

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        totals = {}
        count = 0
        num_batches = len(loader)
        final_accum_size = num_batches % self.grad_accum_steps

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            x = self._get_x(batch)
            with (
                torch.set_grad_enabled(train),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ),
            ):
                losses = self._loss(x, augment=train)
                accum_size = self.grad_accum_steps
                if final_accum_size and i >= num_batches - final_accum_size:
                    accum_size = final_accum_size
                loss = losses["loss"] / accum_size

            if train:
                self.scaler.scale(loss).backward()
                do_step = (i + 1) % self.grad_accum_steps == 0 or i + 1 == num_batches
                if do_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()

            batch_size = x.size(0)
            count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.detach() * batch_size

        if not totals:
            return {}

        names = tuple(totals)
        values = torch.stack(tuple(totals[name] for name in names)) / max(count, 1)
        return dict(zip(names, values.cpu().tolist()))

    def fit(
        self,
        train_loader,
        epochs,
        val_loader=None,
        save_path=None,
        warmup_ratio=0.05,
    ):
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        history = []
        best = float("inf")

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = (
                self._run_epoch(val_loader, train=False)
                if val_loader is not None
                else None
            )
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            metric = (
                val_metrics["loss"]
                if val_metrics is not None
                else train_metrics["loss"]
            )
            msg = (
                f"[{epoch:03d}/{epochs:03d}] "
                f"loss={train_metrics['loss']:.4f} "
                f"token={train_metrics['token']:.4f} "
                f"segment={train_metrics['segment']:.4f} "
                f"wave={train_metrics['wave']:.4f} "
                f"spec={train_metrics['spectrum']:.4f}"
            )

            if val_metrics is not None:
                msg += f" | val={val_metrics['loss']:.4f}"
            print(msg)

            if save_path is not None and metric < best:
                best = metric
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "metric": metric,
                    },
                    save_path,
                )

        return history
