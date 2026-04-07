# ParrotVoice: Mathematical Formulation of the Source Code

This document derives the complete mathematical formulation of the OmniVoice model
directly from the implementation in `omnivoice/model.py`. Sections follow the code's
logical structure: input embedding, forward pass & loss, inference scheduling, and
helper functions.

---

## 1. Model Configuration

The model is parameterized by:

| Symbol                           | Code name                | Description                                          |
| -------------------------------- | ------------------------ | ---------------------------------------------------- |
| $V$                              | `audio_vocab_size`       | Acoustic token vocabulary size (default 1025)        |
| $M$                              | `audio_mask_id`          | Mask token index (default 1024)                      |
| $C$                              | `num_audio_codebook`     | Number of codebook layers (default 8)                |
| $H$                              | `hidden_size`            | LLM hidden dimension                                 |
| $\mathbf{w} = (w_1, \dots, w_C)$ | `audio_codebook_weights` | Per-layer loss weights (default $[8,8,6,6,4,4,2,2]$) |

The normalized layer weights used in the loss are:

$$
\tilde{w}_c = \frac{w_c}{\sum_{c'=1}^{C} w_{c'}}, \quad c = 1, \dots, C
$$

---

## 2. Input Embedding (`_prepare_embed_inputs`)

The model receives a joint input tensor of shape $(B, C, S)$, where $B$ is batch size and $S$ is sequence length. Each position is either a **text token** or an **audio token**, distinguished by the binary audio mask $\mathbf{a} \in \{0,1\}^{B \times S}$.

### 2.1 Text Embedding

For text positions, the standard LLM token embedding $E_{\text{text}} \in \mathbb{R}^{|\mathcal{V}_{\text{text}}| \times H}$ is applied to the first codebook layer (layer index 0):

$$
\mathbf{h}^{\text{text}}_{b,s} = E_{\text{text}}\bigl(\mathbf{x}_{b,0,s}\bigr) \in \mathbb{R}^{H}
$$

### 2.2 Audio Embedding

Each codebook layer $c \in \{0, \dots, C-1\}$ has a dedicated segment within a shared embedding table $E_{\text{audio}} \in \mathbb{R}^{(C \cdot V) \times H}$. Layer $c$ is offset by $c \cdot V$, so the token ID for codebook $c$ at position $s$ is shifted as:

$$
\hat{x}_{b,c,s} = \mathbf{x}_{b,c,s} \cdot \mathbf{a}_{b,s} + c \cdot V
$$

The audio embedding at position $s$ is the **sum over all codebook layers**:

$$
\mathbf{h}^{\text{audio}}_{b,s} = \sum_{c=0}^{C-1} E_{\text{audio}}\!\left(\hat{x}_{b,c,s}\right) \in \mathbb{R}^{H}
$$

### 2.3 Combined Input Embedding

The final per-position embedding selects audio or text based on the mask:

$$
\mathbf{h}_{b,s}
 =
\begin{cases}
\mathbf{h}^{\text{audio}}_{b,s} & \text{if } \mathbf{a}_{b,s} = 1 \\
\mathbf{h}^{\text{text}}_{b,s}  & \text{if } \mathbf{a}_{b,s} = 0
\end{cases}
$$

---

## 3. Forward Pass (`forward`)

### 3.1 Transformer Encoding

The combined embedding $\mathbf{H} \in \mathbb{R}^{B \times S \times H}$ is passed through the bidirectional Transformer backbone $f_\theta$:

$$
\mathbf{Z} = f_\theta(\mathbf{H},\ \mathbf{A}) \in \mathbb{R}^{B \times S \times H}
$$

where $\mathbf{A}$ is the attention mask (either a standard boolean mask or a FlexAttention block mask for sequence-packed training).

### 3.2 Logit Projection

A single linear head $W_{\text{head}} \in \mathbb{R}^{H \times (C \cdot V)}$ projects the hidden states to per-codebook logits:

$$
\mathbf{L}_{\text{flat}} = \mathbf{Z}\, W_{\text{head}}^\top \in \mathbb{R}^{B \times S \times (C \cdot V)}
$$

These are reshaped and transposed to:

$$
\mathbf{L} \in \mathbb{R}^{B \times C \times S \times V}
$$

where $L_{b,c,s,v}$ is the unnormalized log-probability of token $v$ at codebook $c$, position $s$, sample $b$.

---

## 4. Training Loss (`forward`, labels provided)

### 4.1 Per-Token Cross-Entropy

Let $\mathcal{M}$ denote the set of labeled (unmasked target) positions with label index $\neq -100$. The cross-entropy loss at each labeled position $(b, c, s)$ is:

$$
\ell_{b,c,s} = -\log \frac{\exp\!\left(L_{b,c,s,\, y_{b,c,s}}\right)}{\sum_{v=0}^{V-1} \exp\!\left(L_{b,c,s,v}\right)}
$$

where $y_{b,c,s}$ is the ground-truth token. Positions with $y_{b,c,s} = {-100}$ are excluded (`ignore_index=-100`).

### 4.2 Per-Layer Mean Loss

Let $\mathbf{1}_{b,c,s} = \mathbf{1}[y_{b,c,s} \neq -100]$ be the validity indicator. The mean loss for codebook layer $c$ is:

$$
\bar{\ell}_c = \frac{\displaystyle\sum_{b,s} \ell_{b,c,s} \cdot \mathbf{1}_{b,c,s}}{\displaystyle\max\!\left(1,\ \sum_{b,s} \mathbf{1}_{b,c,s}\right)}
$$

### 4.3 Weighted Loss Aggregation

The final scalar training loss is a weighted sum over codebook layers:

$$
\boxed{\mathcal{L} = \sum_{c=1}^{C} \tilde{w}_c \cdot \bar{\ell}_c}
$$

where $\tilde{w}_c$ are the normalized codebook weights defined in Section 1.

---

## 5. Inference: Iterative Masked Decoding (`_generate_iterative`)

### 5.1 Time-Step Schedule (`_get_time_steps`)

A sequence of $N+1$ evenly-spaced time steps is first constructed over $[0, 1]$, then warped by a shift parameter $\tau$:

$$
t_n^{\text{linear}} = \frac{n}{N}, \quad n = 0, 1, \dots, N
$$

$$
\boxed{t_n = \frac{\tau \cdot t_n^{\text{linear}}}{1 + (\tau - 1)\cdot t_n^{\text{linear}}}}
$$

where $\tau = 0.1$ by default (shift parameter `t_shift`). The schedule is monotonically increasing from $t_0=0$ to $t_N=1$, with smaller $\tau$ concentrating more steps near $t=0$ (i.e., unmasking fewer tokens per early step).

### 5.2 Per-Step Unmasking Count

Let $T$ be the target sequence length and $K = T \cdot C$ be the total number of tokens to unmask. At step $n$ ($n = 0, \dots, N-1$), the number of tokens to unmask is:

$$
k_n = \min\!\left(\left\lceil K \cdot (t_{n+1} - t_n) \right\rceil,\ K_{\text{rem}}\right)
$$

where $K_{\text{rem}}$ is the remaining count of masked tokens. On the final step $n = N-1$, all remaining tokens are unmasked: $k_{N-1} = K_{\text{rem}}$.

### 5.3 Classifier-Free Guidance (`_predict_tokens_with_scoring`)

At each decoding step, the model is run in two conditions simultaneously:

- **Conditional** ($\text{cond}$): full context including prompt, text, and reference audio.
- **Unconditional** ($\text{uncond}$): target tokens only, no conditioning context.

Let $\mathbf{L}^{\text{cond}}$ and $\mathbf{L}^{\text{uncond}} \in \mathbb{R}^{C \times T \times V}$ be the respective logit tensors. The guided log-probability is:

$$
\log \tilde{p}_{c,t,v} = \log p^{\text{cond}}_{c,t,v} + \gamma \cdot \left(\log p^{\text{cond}}_{c,t,v} - \log p^{\text{uncond}}_{
    c,t,v}\right)
$$

$$
= (1 + \gamma)\, \log p^{\text{cond}}_{c,t,v} - \gamma\, \log p^{\text{uncond}}_{c,t,v}
$$

where $\gamma$ is `guidance_scale` (default 2.0), and the log-softmax probabilities are:

$$
\log p^{\text{cond}}_{c,t,v} = \log\text{softmax}\!\left(\mathbf{L}^{\text{cond}}_{c,t}\right)_v, \quad \log p^{\text{uncond}}_{c,
t,v} = \log\text{softmax}\!\left(\mathbf{L}^{\text{uncond}}_{c,t}\right)_v
$$

The final guided log-probabilities are renormalized via an additional log-softmax:

$$
\log q_{c,t,v} = \log\text{softmax}\!\left(\log \tilde{p}_{c,t,\cdot}\right)_v
$$

The mask token index $M$ is suppressed at this stage: $\log q_{c,t,M} = -\infty$.

### 5.4 Token Prediction

**Greedy** (`class_temperature` = 0):
$$
\hat{x}_{c,t} = \operatorname{argmax}_{v}\; \log q_{c,t,v}
$$

**Stochastic** (`class_temperature` $> 0$, top-$k$ filtered Gumbel sampling):

Let $\kappa = \lceil \rho \cdot V \rceil$ with ratio $\rho = 0.1$. Define the top-$\kappa$ filtered logits:

$$
\tilde{q}_{c,t,v}
 =
\begin{cases}
\log q_{c,t,v} & \text{if } v \in \text{Top-}\kappa(\log q_{c,t,\cdot}) \\
-\infty & \text{otherwise}
\end{cases}
$$

Then apply Gumbel noise with temperature $\tau_{\text{cls}}$:

$$
g_v \sim \text{Gumbel}(0,1), \quad \hat{x}_{c,t} = \operatorname{argmax}_{v}\left(\frac{\tilde{q}_{c,t,v}}{\tau_{\text{cls}}}
+ g_v\right)
$$

where $g_v = -\log(-\log(u_v + \epsilon) + \epsilon)$, $u_v \sim \mathcal{U}(0,1)$, $\epsilon = 10^{-10}$.

### 5.5 Confidence Score & Layer Penalty

The confidence score for a predicted token at codebook layer $c$, position $t$ is the maximum log-probability:

$$
s_{c,t} = \max_{v}\; \log q_{c,t,v}
$$

A **layer penalty** is subtracted to encourage lower codebook layers (more fundamental acoustic features) to be unmasked first:

$$
\tilde{s}_{c,t} = s_{c,t} - c \cdot \lambda
$$

where $\lambda$ is `layer_penalty_factor` (default 5.0) and $c \in \{0, \dots, C-1\}$ is the zero-indexed codebook layer.

### 5.6 Position Selection via Gumbel Sampling

To introduce stochasticity in position selection, Gumbel noise with temperature $\tau_{\text{pos}}$ is added to the penalized scores:

$$
\hat{s}_{c,t} = \frac{\tilde{s}_{c,t}}{\tau_{\text{pos}}} + g_{c,t}, \quad g_{c,t} \sim \text{Gumbel}(0,1)
$$

where $\tau_{\text{pos}}$ is `position_temperature` (default 5.0). Positions that are **already unmasked** are excluded by setting $\hat{s}_{c,t} = -\infty$ for those positions.

### 5.7 Top-$k$ Position Unmasking

The $k_n$ positions with the highest scores are selected for unmasking at step $n$:

$$
\mathcal{U}_n = \operatorname{Top-}k_n\!\left\{\hat{s}_{c,t} : x_{c,t} = M\right\}
$$

For each $(c, t) \in \mathcal{U}_n$, the token is set to the predicted value:

$$
x_{c,t} \leftarrow \hat{x}_{c,t}
$$

---

## 6. Input Construction for Inference (`_prepare_inference_inputs`)

### 6.1 Token Sequence Layout

The full conditional input sequence is formed by concatenating four segments along the temporal axis:

$$
\mathbf{x}^{\text{cond}} = \bigl[\underbrace{\mathbf{x}^{\text{style}}}_{N_1}\ \|\ \underbrace{\mathbf{x}^{\text{text}}}_{N_2}\ \|
\ \underbrace{\mathbf{x}^{\text{ref}}}_{T_p}\ \|\ \underbrace{\mathbf{x}^{\text{target}}}_{T}\bigr] \in \mathbb{Z}^{C \times S}
$$

where $S = N_1 + N_2 + T_p + T$ and:

- $\mathbf{x}^{\text{style}}$: encoded style tokens (`<|denoise|>`, `<|lang_start|>…<|lang_end|>`, `<|instruct_start|>…<|instruct_end|>`)
- $\mathbf{x}^{\text{text}}$: encoded transcript (`<|text_start|>…<|text_end|>`)
- $\mathbf{x}^{\text{ref}} \in \{0,\dots,V-1\}^{C \times T_p}$: reference audio tokens (omitted if no prompt)
- $\mathbf{x}^{\text{target}} = M \cdot \mathbf{1}^{C \times T}$: all-mask target tokens

### 6.2 Audio Mask

The binary audio mask $\mathbf{a} \in \{0,1\}^S$ marks positions belonging to reference or target audio:

$$
a_s
 =
\begin{cases}
1 & \text{if } s \geq N_1 + N_2 \\
0 & \text{otherwise}
\end{cases}
$$

### 6.3 Unconditional Input

The unconditional input contains only the target segment (target tokens only, no conditioning):

$$
\mathbf{x}^{\text{uncond}} = \mathbf{x}^{\text{target}} \in \mathbb{Z}^{C \times T}
$$

---

## 7. Volume Normalization (`_post_process_audio`)

Let $\rho_{\text{ref}}$ be the RMS of the reference audio waveform. If the reference audio is quiet ($\rho_{\text{ref}} < 0.1$), the generated audio $\hat{\mathbf{y}}$ is rescaled:

$$
\hat{\mathbf{y}} \leftarrow \hat{\mathbf{y}} \cdot \frac{\rho_{\text{ref}}}{0.1}
$$

When no reference audio is provided (voice design mode), peak normalization is applied:

$$
\hat{\mathbf{y}} \leftarrow \frac{\hat{\mathbf{y}}}{\|\hat{\mathbf{y}}\|_\infty} \cdot 0.5
$$

Reference audio itself is clipped to a normalized level before tokenization: if $0 < \rho_{\text{ref}} < 0.1$,

$$
\mathbf{y}_{\text{ref}} \leftarrow \mathbf{y}_{\text{ref}} \cdot \frac{0.1}{\rho_{\text{ref}}}
$$

---

## 8. Duration Estimation and Speed Control

### 8.1 Token Count Estimation

The estimated number of target audio tokens $\hat{T}$ is produced by a rule-based duration estimator. Given a reference text of length $|r|$ characters mapped to $T_p$ reference audio frames, the per-character rate is:

$$
\rho_{\text{char}} = \frac{T_p}{|r|}
$$

The raw target token count is:

$$
\hat{T} = \left\lfloor \rho_{\text{char}} \cdot |t| \right\rfloor
$$

where $|t|$ is the character length of the target text.

### 8.2 Speed Adjustment

Given a speed factor $\alpha > 0$:

$$
T = \max\!\left(1,\ \left\lfloor \frac{\hat{T}}{\alpha} \right\rfloor\right)
$$

Values $\alpha > 1$ produce faster (shorter) speech; $\alpha < 1$ produces slower (longer) speech.

### 8.3 Duration Override

When an explicit duration $d$ (in seconds) is specified, the target token count is set directly from the audio tokenizer frame rate $f_r$:

$$
T = \max\!\left(1,\ \lfloor d \cdot f_r \rfloor\right)
$$

The effective speed ratio for chunked generation is then back-calculated as:

$$
\alpha_{\text{eff}} = \frac{\hat{T}}{T}
$$

---

## 9. Sequence-Packed Attention Mask (`_mask_mod_packed`)

For efficient training with sequence packing, a document-aware causal-style mask is applied. Each position $s$ in the packed sequence is assigned a document ID $\text{doc}(s)$. The attention mask allows position $q$ to attend to position $k$ only if they belong to the same document:

$$
A_{q,k}
 =
\begin{cases}
1 & \text{if } \text{doc}(q) = \text{doc}(k) \\
0 & \text{otherwise}
\end{cases}
$$

Since OmniVoice uses a **bidirectional** Transformer, there is no causal constraint — the full within-document context is available at every position.

---

## Summary Table

| Component                   | Formula                                                                      | Location                       |
| --------------------------- | ---------------------------------------------------------------------------- | ------------------------------ |
| Normalized codebook weights | $\tilde{w}_c = w_c / \sum_{c'} w_{c'}$                                       | `__init__`                     |
| Audio embedding offset      | $\hat{x}_{b,c,s} = x_{b,c,s} \cdot a_{b,s} + c V$                            | `_prepare_embed_inputs`        |
| Summed audio embedding      | $\mathbf{h}^{\text{audio}}_{b,s} = \sum_c E_{\text{audio}}(\hat{x}_{b,c,s})$ | `_prepare_embed_inputs`        |
| Per-token cross-entropy     | $\ell_{b,c,s} = -\log p_{b,c,s,y_{b,c,s}}$                                   | `forward`                      |
| Weighted layer loss         | $\mathcal{L} = \sum_c \tilde{w}_c \bar{\ell}_c$                              | `forward`                      |
| Time-step warp              | $t_n = \tau t_n^{\text{lin}} / (1 + (\tau-1)t_n^{\text{lin}})$               | `_get_time_steps`              |
| CFG log-prob                | $\log q = \text{logsoftmax}((1{+}\gamma)\log p^c - \gamma \log p^u)$         | `_predict_tokens_with_scoring` |
| Layer penalty               | $\tilde{s}_{c,t} = s_{c,t} - c\lambda$                                       | `_generate_iterative`          |
| Position Gumbel sampling    | $\hat{s}_{c,t} = \tilde{s}_{c,t}/\tau_{\text{pos}} + g_{c,t}$                | `_generate_iterative`          |
| Volume normalization        | $\hat{\mathbf{y}} \leftarrow \hat{\mathbf{y}} \cdot \rho_{\text{ref}}/0.1$   | `_post_process_audio`          |
| Speed control               | $T = \max(1, \lfloor \hat{T}/\alpha \rfloor)$                                | `_estimate_target_tokens`      |
| Packed attention            | $A_{q,k} = \mathbf{1}[\text{doc}(q) = \text{doc}(k)]$                        | `_mask_mod_packed`             |
