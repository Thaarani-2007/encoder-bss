# Bibliography & Resources — 2-Person BSS

Organized by category. ⭐ = essential reading.

---

## Surveys & Tutorials

| Resource | Link | Notes |
|----------|------|-------|
| ⭐ Advances in Speech Separation (2025) | https://arxiv.org/html/2508.10830v1 | Most comprehensive recent survey; fair benchmarking |
| Speech Separation DNN Survey (2025) | https://doi.org/10.3390/bdcc9110289 | Evolution U-Net→TasNet→Transformer→Mamba |
| Speech Separation Paper Tutorial | https://github.com/JusperLee/Speech-Separation-Paper-Tutorial | Leaderboards, recipes, dataset links |
| Fair benchmark portal | https://cslikai.cn/Speech-Separation-Paper-Tutorial/ | Unified evaluation (from 2025 survey) |
| Wang & Chen (2018) foundations | https://arxiv.org/abs/1708.03612 | Classic overview |
| ILRMA review (ICA+NMF convergence) | https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/review-of-blind-source-separation-methods-two-converging-routes-to-ilrma-originating-from-ica-and-nmf/83175898245678E77B464369563ABB89 | Classical multi-channel BSS |
| SNN and Sound review | https://pmc.ncbi.nlm.nih.gov/articles/PMC11362401/ | SNN for audio broadly |

---

## Classical BSS

| Paper | Link | Key takeaway |
|-------|------|--------------|
| ICA cocktail party limitations | https://towardsdatascience.com/ica-and-the-real-life-cocktail-party-problem-6375ba35894b/ | Convolutive mixtures break naive ICA |
| Convolutive ICA unifying view | https://arxiv.org/abs/2207.13934 | FD-ICA, IVA, TRINICON relations |
| Multistage ICA for reverb | https://naist.repo.nii.ac.jp/records/4576 | FD-ICA → TD-ICA pipeline |

---

## Foundational Deep Learning

| Paper | Link | SI-SNRi / Notes |
|-------|------|-----------------|
| ⭐ Deep Clustering | https://arxiv.org/abs/1607.02173 | ~10 dB; embedding + clustering |
| ⭐ PIT | https://arxiv.org/abs/1607.00325 | Permutation invariant training |
| Deep Attractor Network | https://pmc.ncbi.nlm.nih.gov/articles/PMC5805382/ | Attractor-based masks |
| uPIT-BLSTM | https://arxiv.org/abs/1708.09588 | Utterance-level PIT |
| TasNet | https://arxiv.org/abs/1711.00541 | Time-domain separation |
| ⭐ Conv-TasNet | https://arxiv.org/abs/1809.07454 | ~15 dB; TCN separator |
| Empirical Conv-TasNet study | https://doi.org/10.1109/icassp40776.2020.9054721 | Encoder matters; generalization hard |

---

## Dual-Path & Transformer Era

| Paper | Link | SI-SNRi / Notes |
|-------|------|-----------------|
| ⭐ DPRNN | https://arxiv.org/abs/1910.06379 | ~18–20 dB; dual-path foundation |
| DPTNet | https://arxiv.org/abs/2007.03992 | Transformer dual-path |
| ⭐ SepFormer | https://arxiv.org/abs/2106.04624 | ~22 dB; ICASSP 2021 |
| ⭐ **SepReformer** | https://arxiv.org/abs/2406.05983 | **~25.1 dB WSJ0-2mix**; NeurIPS 2024; asymmetric encoder-decoder |
| Self-attention exploration | https://arxiv.org/abs/2202.02884 | Reformer vs SepFormer efficiency |
| WaveSplit | https://arxiv.org/abs/2006.13009 | Speaker embeddings in separation |

---

## T-F Domain & SOTA Class

| Paper | Link | SI-SNRi / Notes |
|-------|------|-----------------|
| ⭐ TF-GridNet | https://arxiv.org/abs/2209.03952 | 23.4 dB; complex spectral mapping |
| MossFormer / MossFormer2 | https://arxiv.org/abs/2208.11857 | Convolution-augmented transformer |
| ⭐ SepTDA | https://arxiv.org/abs/2401.12473 | 24.0 dB; unknown # speakers |
| SFSRNet | https://github.com/arda-num/SFSRNet | 24.0 dB |
| QDPN | ICLR 2022 | 23.6 dB; 200M params |

---

## Mamba / State Space Models

| Paper | Link | SI-SNRi / Notes |
|-------|------|-----------------|
| ⭐ DPMamba | https://arxiv.org/abs/2403.18257 | Up to 24.4 dB; dual-path Mamba |
| SPMamba | https://arxiv.org/abs/2404.02063 | TF-GridNet + Mamba; reverb gains |
| SepMamba | https://arxiv.org/abs/2410.20997 | U-Net Mamba; efficient |
| Mamba-TasNet / Speech Slytherin | https://arxiv.org/abs/2407.09732 | Single-path; efficiency analysis |

---

## JEPA & Self-Supervised (Non-Separation)

| Paper | Link | Notes |
|-------|------|-------|
| Stem-JEPA | https://arxiv.org/abs/2408.02514 | Musical stem compatibility, not waveform sep |
| AudioSlots | HuggingFace daily papers | Slot-based generative sep; proof of concept |

---

## Unsupervised & Generalization

| Paper | Link | Notes |
|-------|------|-------|
| ⭐ MixIT | https://arxiv.org/abs/2006.12701 | Mixtures of mixtures; no clean sources |
| Unsupervised deep clustering (spatial) | https://www.mit.edu/~paris/pubs/tzinis-icassp2019.pdf | Multi-channel cues |
| ⭐ Generalization study (AC-SIM) | https://arxiv.org/abs/2408.16126 | Real-world sim pipeline; **primary project paper** |
| ⭐ SepReformer | https://arxiv.org/abs/2406.05983 | Clamped PIT SI-SNR training; **adopted in v2+** |
| Monaural anechoic → reverberant | https://arxiv.org/abs/2111.07578 | Anechoic SOTA ≠ reverb SOTA |
| Data sampling / TSL limits | https://arxiv.org/abs/2304.07142 | Fair Conv-TasNet vs SepFormer comparison |

---

## SNN

| Paper | Link | Notes |
|-------|------|-------|
| Hybrid ANN-SNN speech enhancement | https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1567347/full | Conv-TasNet → SNN conversion |
| Progressive tandem SNN | https://arxiv.org/abs/2007.01204 | Dilated conv SNN for separation |
| Density Networks (bio-inspired) | CCN 2024 | Unsupervised; not DL-SOTA |
| Spike-based audio enhancement thesis | https://repository.rit.edu/cgi/viewcontent.cgi?article=13626 | Denoising focus |

---

## Datasets

| Dataset | Link | Description |
|---------|------|-------------|
| ⭐ WSJ0-2mix | https://www.merl.com/demos/deep-clustering | 8 kHz, 2 speakers, anechoic, full overlap |
| WSJ0-3mix | Same family | 3 speakers |
| Libri2Mix / Libri3Mix | https://github.com/JorisCos/LibriMix | 16 kHz, LibriSpeech-based |
| WHAM! | https://wham.whisper.ai/ | Noise added to wsj0-2mix |
| ⭐ WHAMR! | https://arxiv.org/abs/1910.10279 | Noise + reverberation |
| Echo2Mix | SPMamba paper | LibriSpace reverberant |
| REAL-M | Real recordings | In-the-wild evaluation |

---

## GitHub Repositories & Toolkits

| Repo | Link | Contents |
|------|------|----------|
| ⭐ Asteroid | https://github.com/asteroid-team/asteroid | Conv-TasNet, DPRNN, recipes |
| ⭐ SpeechBrain | https://github.com/speechbrain/speechbrain | SepFormer, training recipes |
| ⭐ **SepReformer** | https://github.com/dmlguq456/SepReformer | Official NeurIPS 2024 code; PIT_SISNR_time loss |
| Mamba-TasNet | https://github.com/xi-j/Mamba-TasNet | DPMamba, Mamba-TasNet official |
| ESPnet separation | https://github.com/espnet/espnet | TF-GridNet recipes |
| Conv-TasNet (original) | https://github.com/kaituoxu/Conv-TasNet | Historical reference |
| DPRNN PyTorch | https://github.com/ShiZiqiang/dual-path-RNNs-DPRNNs-based-speech-separation | Educational |
| Speech Separation Tutorial | https://github.com/JusperLee/Speech-Separation-Paper-Tutorial | Paper list + benchmarks |
| ClearerVoice / MossFormer2 | https://github.com/modelscope/ClearerVoice-Studio | Recent SOTA models |
| WeSep | Referenced in 2025 survey | Toolkit |
| state-spaces/mamba | https://github.com/state-spaces/mamba | Core Mamba implementation |

---

## Pretrained Models (HuggingFace)

| Model | Link | Metric |
|-------|------|--------|
| SepFormer WSJ0-2mix | https://huggingface.co/speechbrain/sepformer-wsj02mix | ~22.4 dB SI-SNR |
| SepFormer WHAMR | https://huggingface.co/speechbrain/sepformer-whamr | Reverberant |
| SepFormer Libri2Mix | SpeechBrain collection | 16 kHz |

---

## Project Paper Guide

For how Chen (AC-SIM), SepFormer, and SepReformer map to our code and configs, see **[PAPERS-AND-PRACTICES.md](./PAPERS-AND-PRACTICES.md)**.

---

## Benchmark Leaderboards

| Source | Link |
|--------|------|
| Papers With Code WSJ0-2mix | https://paperswithcode.com/sota/speech-separation-on-wsj0-2mix |
| SOTA2 leaderboard | https://www.sota2.com/research/sota/speech-separation-on-wsj0-2mix-anechoic-clean-mixture-test |
