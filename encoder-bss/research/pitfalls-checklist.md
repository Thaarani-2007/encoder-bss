# BSS Planning Pitfalls Checklist

Use this before choosing an architecture or claiming SOTA parity. Each item is a common failure mode documented in the literature.

---

## A. Problem & Data Assumptions

- [ ] **Is the mixture instantaneous or convolutive?** Single-channel ICA/NMF fail on reverberant convolutive mixtures without heavy preprocessing.
- [ ] **Is the speaker count fixed at 2?** PIT-based fixed-output models fail gracefully only if C=2 at test time; use attractor/recursive models for unknown C.
- [ ] **What is the overlap pattern?** WSJ0-2mix uses full overlap; real dialogues have turn-taking, partial overlap, and silence — SI-SDR loss under-optimizes low-overlap regions.
- [ ] **Sample rate match?** WSJ0-2mix is 8 kHz; Libri2Mix is 16 kHz. Mixing pretrained models across rates fails.
- [ ] **Do you have isolated source ground truth?** Supervised PIT requires clean sources; unsupervised (MixIT) trades ~2–4 dB and risks over-separation.

## B. Training Configuration (Often Invalidates Paper Comparisons)

- [ ] **Dynamic mixing (DM)?** Most 23+ dB results use on-the-fly remixing; without DM expect ~1–3 dB drop.
- [ ] **Training signal length (TSL) limit?** SepFormer is sensitive above ~3.36s; Conv-TasNet is more invariant. Mismatched TSL makes cross-paper comparison unfair.
- [ ] **Random vs fixed crop start?** WSJ0-derived corpora benefit from random cropping; fixed start can inflate or deflate scores.
- [ ] **Loss function?** SI-SDR in time domain is standard for anechoic; reverberant settings may favor magnitude-only losses.
- [ ] **Mixture consistency loss?** TF-GridNet adds ∑ŝᵢ ≈ x constraint — omitting it can hurt additivity.

## C. Architecture-Specific Traps

### Conv-TasNet
- [ ] Receptive field (~1.5s) limits very long-context modeling; acceptable for 2-speaker anechoic.
- [ ] Encoder/decoder upgrades (deep nonlinear) help but don't fix domain shift alone.
- [ ] Cross-dataset generalization remains weak despite strong WSJ0-2mix numbers.

### SepFormer / Transformers
- [ ] Quadratic memory in sequence length — chunking/dual-path required.
- [ ] Gains on anechoic WSJ0-2mix may **not** transfer to WHAMR!; reverberant SepFormer can be ≈ PIT-BLSTM after tuning.
- [ ] Phase reconstruction jointly with magnitude hurts under reverberation.

### TF-GridNet / T-F Models
- [ ] Sparsity/orthogonality assumptions break under heavy reverberation.
- [ ] Complex spectral mapping + mixture constraint is key to reported gains.
- [ ] Higher MACs than compact Mamba variants.

### Mamba (DPMamba, SPMamba, SepMamba)
- [ ] Efficiency wins appear mainly for **long** sequences; short utterances may not beat Transformer.
- [ ] Ecosystem less mature than SpeechBrain/Asteroid — budget integration time.
- [ ] Bidirectional Mamba needed for offline separation; causal variants trade ~1–2 dB.

### JEPA (Stem-JEPA, etc.)
- [ ] **Not waveform separation** — embedding/retrieval/compatibility, not SI-SDR separation.
- [ ] Limited instrument/speaker vocabulary in training.
- [ ] Could augment a pipeline (retrieval, conditioning) but not replace TasNet-style decoder.

### SNN (Spiking Neural Networks)
- [ ] No competitive SI-SDR on WSJ0-2mix vs ANN baselines.
- [ ] Requires ANN→SNN conversion + surrogate gradients; separation quality gap remains.
- [ ] Justified only for neuromorphic/ultra-low-power deployment research.

### Classical (ICA, NMF, ILRMA)
- [ ] Single-channel underdetermined case: ICA fails without strong priors.
- [ ] NMF static bases miss speech temporal dynamics.
- [ ] ILRMA/IVA need multi-channel for best results; permutation across frequency bins is painful.

## D. Evaluation Traps

- [ ] **SI-SDR vs SI-SNR vs SDR** — not interchangeable; always report improvement (Δ) over mixture.
- [ ] **PESQ/ESTOI/WER downstream** — high SI-SDR doesn't guarantee ASR improvement on separated speech.
- [ ] **In-dataset vs cross-dataset** — model tuned on WSJ0 speech fails on Libri speakers without retraining/DM.
- [ ] **Human listening** — objective metrics miss musical noise and speaker coloration artifacts.

## E. Deployment Traps

- [ ] **Latency:** dual-path chunking adds algorithmic delay; causal mode required for real-time.
- [ ] **Permutation at inference:** output channel order arbitrary; downstream needs speaker ID or diarization.
- [ ] **GPU memory:** full-utterance SepFormer on long meetings OOMs — use chunking with overlap-add.
- [ ] **Domain adaptation:** real meeting room ≠ WSJ0 studio; plan for fine-tune, MixIT, or multi-condition simulation (AC-SIM-style).

---

## Decision Tree (2-Speaker, Single Channel)

```
Start
  │
  ├─ Need unsupervised (no clean sources)?
  │    └─ MixIT / semi-supervised → expect gap vs PIT; watch over-separation
  │
  ├─ Anechoic, clean, benchmark reproduction?
  │    ├─ Fast baseline → Conv-TasNet (Asteroid)
  │    ├─ Strong pretrained → SepFormer (SpeechBrain HF)
  │    └─ Max SI-SDR → TF-GridNet / SepTDA / DPMamba-L + DM
  │
  ├─ Reverberant + noisy (WHAMR!-like)?
  │    └─ Don't trust anechoic SOTA alone; multi-stage (dereverb + sep) or AC-SIM training
  │
  ├─ Real-time / edge?
  │    └─ DPMamba-S / SPMamba / causal SepMamba; avoid full Transformer
  │
  ├─ Research novelty?
  │    ├─ Efficiency → Mamba in dual-path shell
  │    ├─ Unknown # speakers → SepTDA / attractor networks
  │    └─ Avoid → raw JEPA or SNN as primary separator unless that's the research goal
  │
  └─ Multi-mic available?
       └─ Consider beamforming + separation or ILRMA; much easier than monaural
```

---

## Red Flags — Stop and Reconsider

1. Claiming ICA on single-channel reverberant audio will work out of the box.
2. Comparing your model to SepFormer 22 dB without matching DM and TSL protocol.
3. Training only on WSJ0-2mix and deploying in far-field meeting rooms.
4. Choosing JEPA expecting end-to-end 2-speaker waveform output.
5. Expecting SNN to match Conv-TasNet without extensive hybrid ANN pretraining.
6. Ignoring permutation ambiguity in downstream speaker-labeled pipelines.
