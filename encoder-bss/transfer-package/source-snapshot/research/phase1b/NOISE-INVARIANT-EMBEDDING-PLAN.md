# Clean-Teacher / Noisy-Student Embedding Separation

> Status: models, paired data pipeline, and 200k paired cache implemented  
> Date: 2026-09-16  
> Scope: clean/noisy encoder distillation for two-speaker monaural separation

## 1. Proposed idea

The current separator uses a small learned convolutional encoder. The proposed
experiment introduces two stages:

1. Train a clean separation system on clean two-speaker mixtures.
2. Freeze the trained clean encoder as a teacher.
3. Train a larger noisy-mixture encoder as a student.
4. Give the teacher the corresponding clean mixture and the student the noisy
   version of that mixture.
5. Make the student reproduce the teacher's time-aligned embedding.
6. Keep the separator and decoder interface unchanged wherever possible.

The intended result is a noisy encoder whose representation preserves the
speaker-separation information learned from clean mixtures while suppressing
additive-noise variation.

This is a teacher-student / feature-distillation design. It is not merely
pretraining an encoder and then copying its weights.

## 2. Important architecture constraint

The current encoder is:

```python
Conv1d(1, encoder_dim, kernel_size=16, stride=8)
```

It accepts `[B, T]` and returns `[B, encoder_dim, T_enc]`. With the current
settings, the convolution itself has only:

```text
encoder_dim × kernel_size
```

weights: about 2,048 parameters at 128 channels and 4,096 at 256 channels.
This is only the front-end; the separator contains most of the model capacity.

If the separator and decoder remain unchanged, clean and noisy encoders must
both expose the same public shape:

```text
z_clean, z_noisy: [B, C, T_enc]
```

Therefore, “larger noisy encoder” should mean more internal capacity followed
by a projection to the same `C`, not an arbitrary increase of output channels.
A compatible design is:

```text
noisy waveform
  -> wider/deeper Conv1d front-end
  -> normalization and nonlinear blocks
  -> 1x1 projection
  -> [B, C_teacher, T_enc]
  -> existing separator
  -> existing decoder
```

If the noisy encoder instead emits a different channel count, the separator's
first bottleneck and the decoder path must change. That would no longer be a
clean “same separator and decoder” experiment.

## 3. What the teacher should encode

The first version should train the clean teacher on a **clean two-speaker
mixture**, not on isolated speech only:

```text
clean mixture = clean speaker 1 + clean speaker 2
teacher encoder -> mixture embedding -> separator -> two clean sources
```

This matches the downstream input seen by the student. A teacher trained on a
single speaker would provide a speech representation, but it would not
necessarily encode the mixture-level information needed to separate two
speakers.

For this project, “clean” should initially mean **no additive static/event
noise**, while keeping the same sampling rate, duration, speaker split, and
reverberation policy as the target task. A separate anechoic condition can be
added later. Removing both noise and reverberation would create a larger
teacher/student domain gap and would confound the experiment.

The clean target sources remain the same two source waveforms used by PIT.
Noise is not a target speaker.

## 4. Recommended training objective

Let:

```text
z_c = E_clean(x_clean)
z_n = E_noisy(x_noisy)
ŷ_n = D(S(z_n))
```

where `E_clean` is frozen during stage 2, `S` is the separator, and `D` is the
decoder.

The student should not be trained using embedding loss alone. The recommended
stage-2 objective is:

```text
L_total =
    λ_sep  L_PIT-SI-SDR(ŷ_n, s1, s2)
  + λ_feat L_feature(z_n, stopgrad(z_c))
  + λ_rel  L_relation(z_n, stopgrad(z_c))
  + λ_enh  L_optional-clean-reconstruction
```

### 4.1 Feature loss

Start with normalized framewise losses:

```text
L_feature =
    mean(|normalize(z_n) - normalize(z_c)|)
  + β · mean(1 - cosine(z_n, z_c))
```

Use an explicit valid-frame mask. Do not silently compare frames with
different padding or convolution lengths.

The L1/MSE term preserves local values. The cosine term emphasizes direction
and reduces sensitivity to gain differences. A cosine-only loss is risky
because it does not control feature magnitude.

### 4.2 Relational loss

Add only as an ablation if direct feature matching is too restrictive. A
relational loss compares similarities between time frames rather than requiring
the student to reproduce every teacher coordinate:

```text
R(z)[i,j] = cosine(z[...,i], z[...,j])
L_relation = mean(|R(z_n) - R(z_c)|)
```

This is useful when the student has a different internal architecture, but it
is more expensive for long clips. Compute it on sampled frame patches.

### 4.3 Separation loss remains primary

`L_PIT-SI-SDR` must remain active. A student can match the clean encoder while
still producing poor separation if the embedding is not sufficient for the
separator. PIT must use one consistent speaker permutation for all
speaker-indexed losses.

A safe first weighting is:

```text
λ_sep  = 1.0
λ_feat = 0.1
λ_rel  = 0.0 initially
λ_enh  = 0.0 initially
```

Then sweep `λ_feat` over `{0.03, 0.1, 0.3, 1.0}`. These are starting values,
not literature-prescribed constants.

## 5. Teacher and student update policy

### Recommended first run: frozen teacher

1. Train the clean teacher to convergence.
2. Copy its encoder and separator/decoder checkpoint.
3. Freeze the clean encoder with `requires_grad=False`.
4. Set it to evaluation mode.
5. Stop gradients through its embeddings.
6. Train only the noisy student and, initially, the separator/decoder.

A frozen teacher gives a stable target and makes the experiment auditable.
The teacher should not accidentally learn the noisy domain and move the target
during student optimization.

### Optional comparison: EMA teacher

An exponential-moving-average teacher can be tested after the frozen-teacher
baseline. It should not be the default because it can drift from the clean
representation:

```text
teacher <- μ · teacher + (1 - μ) · student
```

Test `μ` values such as 0.99 and 0.999, while retaining PIT separation loss
and embedding-variance diagnostics. An EMA teacher without a grounded
separation objective can collapse or gradually forget the clean space.

## 6. The proposed 70/30 data mixture

The proposed 70% noisy / 30% clean schedule is reasonable as a first
engineering hypothesis:

- noisy batches teach robustness;
- clean batches prevent the student from treating every input as a denoising
  problem and preserve clean separation behavior;
- the clean fraction supplies direct separation supervision without requiring
  the teacher to be perfect.

However, the reviewed literature does **not** establish 70/30 as a universal
optimal ratio. Published systems use staged training, uniform corruption
sampling, smaller supervised fractions, or separate noisy and clean objectives.
Therefore, 70/30 must be measured rather than treated as a fact.

Required ratio ablation:

```text
100/0 clean/noisy
90/10 clean/noisy
70/30 noisy/clean  (proposed starting point; state the convention explicitly)
50/50
0/100 noisy/clean
noisy pretraining -> clean fine-tuning
```

The ratio must be implemented at batch level and logged. A ratio of “70%
noise data” is ambiguous unless it states whether it means:

1. 70% batches contain noisy mixtures;
2. 70% examples are noisy;
3. 70% of the loss weight is assigned to noisy examples; or
4. 70% of the raw source pool contains noise.

The recommended definition is **70% noisy examples and 30% clean examples in
each training epoch**, with both examples retaining their paired clean
reference mixture and speaker targets.

## 7. Data construction

For every clean two-speaker training item, create a paired noisy item:

```text
clean mixture x_c = s1 + s2
noisy mixture x_n = s1 + s2 + noise
teacher target: E_clean(x_c)
student input:  E_noisy(x_n)
separation targets: s1, s2
```

Use the same speaker pair, crop, gains, and reverberation in the pair. Only
the additive noise condition should differ in the first controlled experiment.
This prevents the embedding loss from learning speaker, crop, or RIR
differences instead of noise invariance.

For AC-SIM, create separate manifests rather than relying on a random runtime
flag:

```text
clean_teacher_train
paired_noisy_student_train
clean_validation
noisy_validation_by_SNR
```

Do not mix teacher and student inputs in a cache without storing the pairing
metadata. Record seed, noise identity, SNR, RIR identity, crop offset, and
speaker IDs.

## 8. Alignment requirements

The current unpadded Conv1d has approximately:

```text
T_enc = floor((T - kernel_size) / stride) + 1
```

Teacher and student must use identical:

- sample rate;
- kernel size and stride at the public interface;
- padding and crop length;
- framing convention;
- temporal resampling;
- valid-frame mask.

If the larger student front-end changes temporal resolution, add an explicit
alignment layer:

```text
z_student_internal -> 1x1 projection -> temporal interpolation/pooling -> z_student
```

Never solve a mismatch by silently truncating to the shorter tensor. Log
`T_clean`, `T_noisy`, `T_teacher`, `T_student`, and the number of valid frames.

## 9. Diagnostics required for this project

In addition to D-All and S-All, record:

### Representation diagnostics

- teacher/student cosine similarity;
- normalized feature L1 and MSE;
- per-channel variance;
- fraction of near-zero channels;
- effective rank or singular-value spectrum;
- average pairwise cosine similarity between unrelated examples;
- clean-versus-noisy embedding distance by SNR;
- embedding distance by noise type and reverberation condition.

### Separation diagnostics

- D-All SI-SDRi;
- D-NE, D-NR, D-N;
- S-All and all S splits;
- source-collapse percentage;
- mixture-copy percentage;
- speaker-swap rate;
- clean-input versus noisy-input performance;
- performance with the teacher encoder, student encoder, and frozen separator.

### Scientific controls

At minimum compare:

1. Existing baseline trained from scratch.
2. Clean teacher encoder + original separator/decoder.
3. Student initialized from teacher and trained with PIT only.
4. Student trained with PIT plus feature matching.
5. Student trained with feature matching only.
6. 70/30 ratio versus ratio ablations.

The proposed method only succeeds if it improves noisy D-All without materially
degrading clean D-All and without creating source collapse.

## 10. Main risks and hypotheses

### H1: Clean embeddings contain useful separation information

If the clean teacher reaches good clean D-All, its intermediate embedding
should provide a useful target. This is the central hypothesis.

### H2: Direct mixture-embedding matching may be under-constrained

Many different embeddings can produce the same separated waveform. The student
may match the teacher poorly while still separating well, or match the teacher
well while losing information needed by the separator. This is why both feature
and PIT losses are required.

### H3: Pure feature matching may remove useful noise context

The separator may need to know the noise condition to avoid assigning noise to
speakers. A strict invariant embedding can discard this information. Keep the
student's final embedding speech-focused, but consider a small separate
condition/noise branch later rather than forcing all information into one
embedding.

### H4: A larger student can overfit the teacher shortcut

The student may memorize the clean teacher's limitations or exploit gain and
normalization statistics. Use held-out noise types, unseen SNRs, and clean
validation.

### H5: Clean/noisy temporal alignment can dominate the loss

Noise changes energy but should not change the speech crop. If the pair has
different preprocessing or padding, the embedding loss measures misalignment,
not robustness.

### H6: The 70/30 schedule may be wrong

More noisy data may improve robustness but harm clean separation. More clean
data may preserve separation but fail to remove noise. This must be tested.

## 11. Research evidence

The following sources were reviewed because they contain either clean/noisy
representation matching, teacher-student separation, feature distillation, or
noise-aware separation.

### Directly relevant papers

1. **Learning Speaker Embeddings from Mixtures with Teacher-Student Training**,
   2023.  
   https://arxiv.org/abs/2306.00634  
   Directly supports a clean single-speaker teacher supervising
   mixture-input, framewise speaker embeddings. It reinforces the need for
   explicit PIT/uPIT handling and frame alignment.

2. **SPIRAL: Self-supervised Perturbation-Invariant Representation Learning
   for Speech Pre-Training**, ICLR 2022.  
   https://arxiv.org/abs/2201.10207  
   Clean teacher and perturbed-input student learn similar representations;
   includes teacher updates and noise-robust multi-condition training.

3. **RobustDistiller: Compressing Universal Speech Representations for
   Enhanced Environment Robustness**, ICASSP 2023.  
   https://github.com/Hguimaraes/robustdistiller  
   Feature-denoising distillation matches noisy student hidden states to clean
   teacher states and adds clean waveform/STFT reconstruction.

4. **Ultra Fast Speech Separation Model with Teacher Student Learning**,
   INTERSPEECH 2021.  
   https://www.isca-archive.org/interspeech_2021/chen21l_interspeech.pdf  
   https://github.com/Sanyuan-Chen/CSS_with_TSTransformer  
   Uses layer-wise teacher-student losses, output distillation, and objective
   shifting toward clean targets for speech separation.

5. **Teacher-Student MixIT for Unsupervised and Semi-Supervised Speech
   Separation**, INTERSPEECH 2021.  
   https://www.isca-archive.org/interspeech_2021/zhang21v_interspeech.pdf  
   Supports a fixed teacher, PIT student training, and later distillation, but
   its data proportions do not validate a 70/30 rule.

6. **Noise-Aware Speech Separation with Contrastive Learning**, ICASSP 2024.  
   https://doi.org/10.1109/icassp48485.2024.10448214  
   https://github.com/TzuchengChang/NASS  
   Predicts noise as an additional output and uses patch-wise contrastive
   learning to keep noise out of speaker representations.

7. **An Efficient End-to-End Approach to Noise Invariant Speech Features via
   Multi-Task Learning**, 2024.  
   https://arxiv.org/abs/2403.08654  
   Directly studies clean teacher representations, noisy student inputs,
   denoising reconstruction, and robust latent features.

8. **Audio Barlow Twins**, 2022.  
   https://arxiv.org/abs/2209.14345  
   Provides a non-contrastive redundancy-reduction objective that can prevent
   representation collapse while enforcing augmentation invariance.

9. **Wav2code: Restore Clean Speech Representations via Codebook Lookup for
   Noise-Robust ASR**, 2023/2024.  
   https://arxiv.org/abs/2304.04974  
   Shows that predicting clean representations can avoid distortions caused by
   directly enhancing the waveform; it is ASR-focused, not a separation recipe.

### Verified implementation references

1. Asteroid encoder-masker-decoder separation toolkit:  
   https://github.com/asteroid-team/asteroid

2. Official SPIRAL implementation:  
   https://github.com/huawei-noah/Speech-Backbones/tree/main/SPIRAL

3. RobustDistiller implementation and checkpoints:  
   https://github.com/Hguimaraes/robustdistiller

4. Teacher-student separation implementation:  
   https://github.com/Sanyuan-Chen/CSS_with_TSTransformer

5. NASS implementation:  
   https://github.com/TzuchengChang/NASS

6. SpeechBrain separation and enhancement toolkit:  
   https://github.com/speechbrain/speechbrain

These repositories support modular encoder/separator/decoder design, PIT
separation losses, cosine or feature distillation, clean/noisy augmentation,
and noise-aware auxiliary objectives. None provides this project's exact
clean-mixture teacher plus larger noisy-mixture student configuration, so the
proposed combination remains a project hypothesis.

## 12. Recommended implementation sequence

1. Add an encoder API that can return `[B,C,T_enc]` plus valid lengths.
2. Generate paired clean/noisy mixtures from identical source/crop/RIR draws.
3. Train and fully evaluate the clean teacher.
4. Freeze the teacher and verify its embeddings are stable.
5. Initialize the student from the teacher's public front-end where shapes
   match; add internal capacity and a projection where needed.
6. Train a student with PIT SI-SDR only.
7. Add normalized feature matching with a low weight.
8. Compare 70/30 with the required ratio ablations.
9. Add relational, enhancement, EMA, or NASS-style objectives only after the
   base experiment is stable.
10. Report clean and noisy metrics separately, including collapse and
    embedding diagnostics.

The first scientifically clean result should be the comparison between:

```text
baseline noisy model
teacher-initialized student + PIT
teacher-initialized student + PIT + feature matching
```

No claim that the method works should be made from embedding loss reduction
alone.

## 13. Implemented Conv-TasNet model pair

The implementation is in
`src/bss_audio/models/conv_tasnet_embedding.py`.

### Clean teacher

`CleanConvTasNet` follows the reference Conv-TasNet configuration as closely
as the current codebase allows:

| Paper/reference component | Implementation |
|---|---|
| Learned 1-D encoder | `Conv1d(1, 512, kernel_size=16, stride=8)` |
| Encoder activation | ReLU |
| Bottleneck | 1x1 convolution to 128 channels |
| Temporal separator | Depthwise-separable dilated Conv-TasNet blocks |
| Blocks/repeats | 8 blocks × 3 repeats |
| Hidden channels | 512 |
| Normalization | `GroupNorm(1, C)`, equivalent to gLN for `[B,C,T]` tensors |
| Mask output | Two-source sigmoid masks |
| Decoder | Learned `ConvTranspose1d`, stride 8 |

The clean model is the comparison baseline. It uses no teacher-specific
shortcut in its forward path; it is a normal Conv-TasNet-style separator that
also exposes its encoder embedding.

### Noisy student

`NoisyConvTasNetStudent` keeps the same Conv-TasNet masker and decoder, but
replaces the shallow 4,096-parameter raw 512-channel front-end with:

```text
Conv1d waveform stem -> depthwise-separable temporal blocks -> 1x1 projection
```

The default student front-end uses 192 hidden channels and two temporal blocks,
which is intentionally in the low-hundreds-of-thousands of parameters. It
still outputs 512 channels so the masker/decoder interface is unchanged.

### Deliberate deviations

This is paper-aligned, not a byte-for-byte reproduction:

1. The original Conv-TasNet uses a specific implementation of global layer
   normalization and temporal blocks; this code uses PyTorch `GroupNorm(1,C)`
   as the equivalent per-example channel/time normalization.
2. The original paper's exact mask activation and filterbank variants differ
   across reference configurations; this baseline uses the standard learned
   convolutional filterbank and sigmoid masks.
3. The model is integrated into this repository's 16-kHz, two-speaker,
   AC-SIM/PIT training setup rather than the paper's WSJ0-2mix recipe.
4. The noisy student is a new project model. Its larger front-end is not part
   of standard Conv-TasNet; it is the controlled robustness extension being
   tested.

The clean model should be trained and evaluated before the noisy student.
Comparisons must report parameter counts for the front-end, masker, decoder,
and complete model separately.

## 14. Implemented artifacts and verification

### Model implementation

Implemented in `src/bss_audio/models/conv_tasnet_embedding.py`:

- `CleanConvTasNet`: paper-aligned clean teacher.
- `NoisyConvTasNetStudent`: larger noisy front-end with the same 512-channel
  public embedding interface.
- Shared Conv-TasNet-style learned filterbank, temporal masker, sigmoid masks,
  and transposed-convolution decoder.

The model factory recognizes `convtasnet_clean` and `convtasnet_noisy`.
Compilation and a CUDA forward/backward smoke test passed. The clean model
reported 5.06M total parameters with the paper-scale masker configuration.

### Normalized speech source mirror

Created and independently verified:

```text
E:\bss-audio-data\speech_16k\
E:\bss-audio-data\manifests\speech_split_16k.json
```

The mirror contains 220,981 train/eval files. Every file is mono, 16 kHz,
and preserves the speaker-disjoint split:

```text
train files: 198,763
eval files:   22,318
speakers:       1,283
eval speakers:    128
```

The original speech tree and original caches were not modified.

### Paired cache

The canonical paired cache is:

```text
E:\bss-audio-data\cache\convtasnet_paired_200k\
```

It contains 200,000 clips with global indices `0..199999`. Each `.pt` file
contains:

```text
mixture        noisy-mixture compatibility alias
noisy_mixture  noisy student input
clean_mixture  clean teacher input
target         [2, 80000] clean speaker targets
source1_path
source2_path
index
seed
```

The manifest records 16 kHz, 80,000 samples per clip, the same speaker split,
and the clean/noisy simulation probabilities. The cache merge copied the
verified second part without deleting its source folder. The combined cache
was verified for 200,000 indices and 1,000 sampled paired clips; all sampled
keys, shapes, aliases, and finite-value checks passed.

The current paired cache uses two-speaker examples (`p_speaker2: 1.0`) for
clean/noisy alignment. It does **not** yet implement the proposed 70/30
clean/noisy batch schedule. That schedule belongs to the student training
loader and must be evaluated as an experiment after the paired cache is
available.

### Current scientific status

The data and model infrastructure is ready. As of 2026-09-16 the training stack
is wired: paired cache fields in `CachedACSIMDataset`, `pit_sisnr_distill` in
`acsim_ml.py`, frozen-teacher student training in `train/loop.py`, and Run 7
configs under `configs/convtasnet_*.yaml`. See [RUN7-PLAN.md](./RUN7-PLAN.md).

No clean-teacher or noisy-student **quality claim** has been made yet: full GPU
training, ratio ablations, and 225-clip evaluation remain the next experimental
steps.
