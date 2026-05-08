# CF-VLA

CF-VLA introduces a plug-and-play action expert that replaces the action generation component in flow-based VLA backbones, enabling seamless integration while improving both inference speed and action generation performance without modifying the underlying architecture.

📄 Paper: https://arxiv.org/abs/2604.24622

## Core Information

- Core model implementation: `src/openpi/models_pytorch/cf_vla.py`
- Compatible configuration name: `pi0_2stg_pytorch`
- Training entry point: `scripts/train_pytorch.py`
- One-click launch script: `scripts/run_train_cf_vla.sh`

## Core Implementation Appendix

### Code Map

The reference two-stage policy lives in `src/openpi/models_pytorch/cf_vla.py`, with the compatible configuration name `pi0_2stg_pytorch`.

The implementation is built on top of a $\pi_{0.5}$-style VLA policy, but the coarse-to-fine design is not tied to this specific codebase. It can also be instantiated on other policy frameworks with analogous prefix/suffix action-generation interfaces.

At a high level, a $\pi_{0.5}$-style policy can be understood in three blocks:

1. `embed_prefix` converts image and language inputs into prefix tokens and runs a cached VLM forward pass.
2. `embed_suffix` constructs the action-side tokens consumed by the action expert, injecting timestep information while reusing the prefix context.
3. `forward()` and `sample_actions()` repeatedly call the suffix path while reusing the cached prefix states, so most action-generation logic lives on the suffix side.

CF_VLA keeps this overall scaffold and modifies the places that matter for coarse-to-fine action generation. Compared with a plain $\pi_{0.5}$ policy, CF_VLA adds a Gaussian prediction head for coarse initialization, inserts an explicit coarse stage before the refinement stage inside `forward()` and `sample_actions()`, and exposes phase-specific switches in `model_kwargs` so the same backbone can realize both the Phase I warm-up path and the Phase II coupled path.

Main components:

- **Shared $\pi_{0.5}$ prefix path.**  
  `embed_prefix` builds prefix tokens from images and language. A cached forward through `paligemma_with_expert` yields `past_key_values` for the action expert.

- **Shared $\pi_{0.5}$ suffix path.**  
  `embed_suffix` builds action and time embeddings. `forward_suffix` and `denoise_step` run the expert trunk on suffix tokens while attending to the cached prefix.

- **New Gaussian posterior head.**  
  Mean and log-variance are predicted by `action_out_proj` and `action_out_proj_logvar`, then packed into `DiagonalGaussianDistribution`, which implements `mode()`, `sample()`, and `kl()`.

- **New two-stage control flow.**  
  Phase I and Phase II appear as branches inside `forward()` and `sample_actions()`. Routing is controlled through `model_kwargs`, including:
  - `1stg_loss_type`
  - `1stg_as_noise`
  - `1stg_output_mode`
  - `times_list_train`
  - `times_start_test`

- **Loss composition.**  
  The Phase I log-variance coefficient is implemented by `loss_logvar_weight`. The phase-specific coarse-loss weights $\lambda_{\mathrm{I}}$ and $\lambda_{\mathrm{II}}$ are both implemented by `loss_1stg_weight` in their respective phases.

- **Sampling schedule.**  
  `sample_actions()` first runs a coarse pass near $t=1$, then a fine pass from the coarse output. Step scaling follows the configured refinement start time, for example `times_start_test`.

---

### Phase-to-Code Switch Summary

Once the shared $\pi_{0.5}$ scaffold and the new Gaussian head are fixed, Phase I and Phase II can be read as two straight-line executions of the same backbone. They mainly differ in how the coarse output is supervised and how it seeds the fine stage.

- **Phase I warm-up.**  
  After building the prefix cache once, the code first performs a coarse pass at $t=1$ from a zero initialization. The posterior mean is supervised with MSE, and the log-variance is supervised with a separate coefficient implemented by `loss_logvar_weight`. Then the code performs a second suffix forward at $t=0.1$ to regress the usual refinement target. The whole Phase I coarse loss is weighted by `loss_1stg_weight`.

- **Phase II coupled training.**  
  The code keeps the same prefix/suffix scaffold, but changes the coarse branch into a Gaussian posterior-matching step. The coarse output is trained against a target Gaussian via KL, and a sample from that posterior is directly used to initialize the fine branch. The Phase II coarse loss is also weighted by `loss_1stg_weight`, but with the Phase II-specific value.

This is why the implementation is best understood in four logical pieces:

1. Shared modules.
2. Phase I training and sampling.
3. Phase II training and sampling.
4. Mapping back to the full `forward()` and `sample_actions()` implementation.

---

### Gaussian Posterior Helper and Suffix Forward

The first architectural change beyond a plain $\pi_{0.5}$ action head is that the suffix path predicts both mean and log-variance, instead of only a deterministic action-direction vector.

The helper below packages this output as a diagonal Gaussian posterior. This posterior is reused by both the coarse stage and the refinement stage.

```python
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -5.0, 20.0)
        # std, var, sample(), kl(other), nll(...), mode() elided

    def mode(self) -> torch.Tensor:
        return self.mean


# forward_suffix: suffix expert -> linear heads -> Gaussian parameters
suffix_out = hidden_states[:, -self.config.action_horizon :]

v_t = torch.cat(
    [
        self.action_out_proj(suffix_out),
        self.action_out_proj_logvar(suffix_out),
    ],
    dim=-1,
)

return DiagonalGaussianDistribution(v_t)
```

---

### Phase I: Warm-up Training and Sampling

#### Training: Coarse MSE Warm-up Plus Fine Regression

Phase I training first builds the prefix cache once, then runs a coarse pass at $t=1$ from a zero state.

The coarse branch regresses:

```text
u_t = noise - action
```

In the default endpoint setting, the coarse input is zero, so this becomes:

```text
u_t = 0 - action
```

The coarse branch also predicts log-variance. The target log-variance is constructed from `noise_var`, and the matching term is weighted by `loss_logvar_weight`.

After that, the fine branch performs a second suffix forward at $t=0.1$ and regresses the usual refinement target.

The total Phase I objective is:

```text
fine-stage MSE + loss_1stg_weight * coarse-stage loss
```

Distilled code path:

```python
loss_1stg_weight = ...   # Phase I coarse-loss weight
noise_var = ...          # target variance
loss_logvar_weight = ... # log-variance loss weight

images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
    observation,
    train=True,
)

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images,
    img_masks,
    lang_tokens,
    lang_masks,
)

prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

past_key_values = self._apply_checkpoint(
    forward_func,
    prefix_embs,
    prefix_att_2d_masks_4d,
    prefix_position_ids,
)

# Coarse branch
one_noise = torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)
one_time = torch.ones(actions.shape[0], dtype=torch.float32, device=actions.device)

x_t = one_noise
one_u_t = one_noise - actions

posterior = self.forward_suffix(
    state,
    x_t,
    one_time,
    prefix_pad_masks,
    past_key_values,
    None,
)

mse_loss = F.mse_loss(one_u_t, posterior.mode(), reduction="none")

target_logvar = torch.log(
    torch.tensor(noise_var, device=one_u_t.device, dtype=one_u_t.dtype)
) * torch.ones_like(one_u_t)

mse_loss = mse_loss + loss_logvar_weight * F.mse_loss(
    target_logvar,
    posterior.logvar,
    reduction="none",
)

loss_1stg = mse_loss.mean(dim=[1, 2])

# Fine branch
noise = self.sample_noise(actions.shape, actions.device)

time = torch.full(
    (actions.shape[0],),
    0.1,
    dtype=torch.float32,
    device=actions.device,
)

time_expanded = time[:, None, None]

x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions

posterior = self.forward_suffix(
    state,
    x_t,
    time,
    prefix_pad_masks,
    past_key_values,
    None,
)

v_t = posterior.mode()
action_loss = F.mse_loss(u_t, v_t, reduction="none")

combined_loss = action_loss.mean() + loss_1stg_weight * loss_1stg

return combined_loss, {
    "action_loss": action_loss.mean().item(),
    "loss_1stg": loss_1stg.mean().item(),
}
```

#### Sampling: One Coarse Step and One Fine Step

At inference time, the prefix cache is built once. Each stage calls `denoise_step`, which runs the action expert with frozen prefix KV cache.

The default sampling process is:

1. Start from `x_t = 0` at `t = 1`.
2. Run one coarse Euler step.
3. Treat the result as the coarse initialization.
4. Run one fine refinement step at `t = 0.1`.

Distilled code path:

```python
images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
    observation,
    train=False,
)

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images,
    img_masks,
    lang_tokens,
    lang_masks,
)

prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)

bsize = observation.state.shape[0]
actions_shape = (
    bsize,
    self.config.action_horizon,
    self.config.action_dim,
)

# Coarse step
one_noise = torch.zeros(actions_shape, dtype=torch.float32, device=device)

dt = torch.tensor(-1.0, dtype=torch.float32, device=device)
x_t = one_noise

time = torch.tensor(1.0, dtype=torch.float32, device=device)
expanded_time = time.expand(bsize)

posterior = self.denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    expanded_time,
)

x_t = x_t + dt * posterior.mode()
noise = x_t

# Fine step
dt = torch.tensor(-1.0, dtype=torch.float32, device=device)
x_t = noise

time = torch.tensor(0.1, dtype=torch.float32, device=device)
dt = dt * 0.1

x_t *= 0.9

expanded_time = time.expand(bsize)

posterior = self.denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    expanded_time,
)

x_t = x_t + dt * posterior.mode()

return x_t
```

---

### Phase II: KL Coarse Stage and Coupled Fine Branch

#### Training: KL on Coarse Posterior, Then Coarse Output Seeds the Fine Branch

Phase II keeps the same overall flow as Phase I, but changes the semantics of the coarse branch.

The coarse pass at $t=1$ is interpreted as a Gaussian posterior. It is matched through KL to the target Gaussian:

```text
N(u_t, noise_var * I)
```

where:

```text
u_t = 0 - action
```

Then a sample from the predicted posterior is used to initialize the fine branch. The fine branch regresses:

```text
v_t -> noise - action
```

where `noise` is constructed from the sampled coarse state rather than from an independent Gaussian sample.

Distilled code path:

```python
loss_1stg_weight = ...  # Phase II coarse-loss weight
noise_var = ...         # KL target variance

images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
    observation,
    train=True,
)

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
    images,
    img_masks,
    lang_tokens,
    lang_masks,
)

prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

past_key_values = self._apply_checkpoint(
    forward_func,
    prefix_embs,
    prefix_att_2d_masks_4d,
    prefix_position_ids,
)

# Coarse branch
one_noise = torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)
one_time = torch.ones(actions.shape[0], dtype=torch.float32, device=actions.device)

x_t = one_noise
one_u_t = one_noise - actions

posterior = self.forward_suffix(
    state,
    x_t,
    one_time,
    prefix_pad_masks,
    past_key_values,
    None,
)

other = torch.cat(
    [
        one_u_t,
        torch.log(
            torch.tensor(noise_var, device=one_u_t.device, dtype=one_u_t.dtype)
            * torch.ones_like(one_u_t)
        ),
    ],
    dim=-1,
)

loss_1stg = posterior.kl(DiagonalGaussianDistribution(other))

# Fine branch initialized from the coarse posterior
noise = one_noise - posterior.sample()

time = torch.full(
    (actions.shape[0],),
    0.1,
    dtype=torch.float32,
    device=actions.device,
)

x_t = noise
u_t = noise - actions

posterior = self.forward_suffix(
    state,
    x_t,
    time,
    prefix_pad_masks,
    past_key_values,
    None,
)

v_t = posterior.mode()
action_loss = F.mse_loss(u_t, v_t, reduction="none")

combined_loss = action_loss.mean() + loss_1stg_weight * loss_1stg

return combined_loss, {
    "action_loss": action_loss.mean().item(),
    "loss_1stg": loss_1stg.mean().item(),
}
```

#### Sampling

Phase II uses the same high-level two-step inference schedule as Phase I:

1. One coarse Euler step at $t=1$.
2. One fine refinement step at $t=0.1$.

The main difference from Phase I is not the inference structure itself, but the training objective: Phase II couples the fine initializer to the coarse posterior.

```python
# Prefix cache is identical to the Phase I sampling path.

bsize = observation.state.shape[0]
actions_shape = (
    bsize,
    self.config.action_horizon,
    self.config.action_dim,
)

# Coarse step
one_noise = torch.zeros(actions_shape, dtype=torch.float32, device=device)

dt = torch.tensor(-1.0, dtype=torch.float32, device=device)
x_t = one_noise

time = torch.tensor(1.0, dtype=torch.float32, device=device)
expanded_time = time.expand(bsize)

posterior = self.denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    expanded_time,
)

x_t = x_t + dt * posterior.mode()
noise = x_t

# Fine step
dt = torch.tensor(-1.0, dtype=torch.float32, device=device)
x_t = noise

time = torch.tensor(0.1, dtype=torch.float32, device=device)
dt = dt * 0.1

expanded_time = time.expand(bsize)

posterior = self.denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    expanded_time,
)

x_t = x_t + dt * posterior.mode()

return x_t
```

---

### Mapping to the Full `forward()` and `sample_actions()` Methods

The production implementation interleaves the above logic with several compatibility branches. The phase-specific paths correspond to the following entries in `model_kwargs`:

- `1stg_loss_type`: selects the coarse objective.
  - `mse`: default Phase I warm-up path.
  - `kl`: default Phase II path.
  - `nll`: optional Phase II ablation.

- `1stg_as_noise`: whether the coarse output is routed into the fine initializer.

- `1stg_output_mode`: whether to use `posterior.mode()` or `posterior.sample()` when forming the fine initializer.

- `times_t1`: optional rescale of the coarse terminal time.

- `times_start_test`: refinement start time and step-size scaling at inference.

- `times_list_train`: candidate fine times sampled during training.

- `noisy_actions`: clean coarse state versus diffusion-style mixing for the fine pass.

- `flow_num`: number of fine samples averaged per optimization step.

In short, the full implementation can be read as a $\pi_{0.5}$-style prefix/suffix policy with three CF_VLA-specific additions:

1. a Gaussian posterior head,
2. a coarse endpoint stage,
3. a single-step fine refinement stage initialized from the coarse output.


## Environment Setup

> Use the standalone environment file `environment.yml` to create the `cf_vla` conda environment for GPU reproduction.

```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate cf_vla

# 3. Install the project package
pip install -e .

# 4. Install the openpi client
pip install -e packages/openpi_client
```

## Optional Benchmark Dependencies

```bash
pip install -e ".[bench-libero]"
pip install -e ".[bench-calvin]"
```

## Training Example

```bash
bash scripts/run_train_cf_vla.sh \
  configs/pi05_2stg_pytorch_delta_actions/Cf_vla_libero.yaml
```

## Inference and Evaluation Example

Start the policy server, then run the evaluation client:

```bash

# Server

python scripts/serve_policy.py --env LIBERO

# Client

python examples/libero/main.py
```


## 📄 Citation

If you find this work useful, please cite:

Du et al., CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies, arXiv preprint arXiv:2604.24622, 2026.
📄 Paper: https://arxiv.org/abs/2604.24622

```bibtex
@article{du2026cfvla,
  title={CF-VLA: Efficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies},
  author={Du, Fan and Yan, Feng and Wu, Jianxiong and Xu, Xinrun and Zhang, Weiye and Wang, Weinong and Guo, Yu and Qian, Bin and He, Zhihai and Wang, Fei and Yang, Heng},
  journal={arXiv preprint arXiv:2604.24622},
  year={2026}
}


## License

- Code license: MIT. See `LICENSE`.
- Third-party dependency notes: `THIRD_PARTY_NOTICES.md`
- Model weights and datasets may have separate licenses. Please verify them separately before release.
