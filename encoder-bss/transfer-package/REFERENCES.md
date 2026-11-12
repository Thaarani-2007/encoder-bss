# References

Only primary or official sources are listed. No citation is invented.

1. Luo, Yi, and Mesgarani, “Dual-path RNN: efficient long sequence modeling for
   separation,” arXiv:1910.06379. https://arxiv.org/abs/1910.06379  
   Decision mapping: dual-path RNN structure, chunking and intra/inter recurrence.
2. SpeechBrain official DPRNN recipe:
   https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/hparams/dprnn.yaml  
   Decision mapping: reference encoder dimensions, six-layer recipe, PIT SI-SNR and
   plateau scheduling comparison.
3. SpeechBrain official dual-path implementation:
   https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py  
   Decision mapping: implementation cross-check for dual-path segmentation and blocks.
4. Subakan et al., “Attention is all you need in speech separation,” arXiv:2010.13154.
   https://arxiv.org/abs/2010.13154  
   Decision mapping: SepFormer architectural reference; this repo's model is explicitly
   only SepFormer-inspired.
5. AC-SIM / AC-SIM-ML primary paper, arXiv:2408.16126.
   https://arxiv.org/abs/2408.16126  
   Decision mapping: simulation, multi-loss framing, evaluation comparison, and reported
   paper reference values recorded by this repository.
6. PyTorch official documentation: https://pytorch.org/docs/stable/  
   Decision mapping: model, DataLoader, autocast, checkpoint and gradient APIs.
7. Weights & Biases official documentation: https://docs.wandb.ai/  
   Decision mapping: run lifecycle, metric axes and artifact logging.

8. Yu et al., “Permutation Invariant Training of Deep Models for Speaker-Independent
   Multi-Talker Speech Separation,” arXiv:1607.00325.
   https://arxiv.org/abs/1607.00325  
   Decision mapping: PIT assignment handling and the two-speaker permutation objective.
9. Luo and Mesgarani, “Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for
   Speech Separation,” arXiv:1809.07454.
   https://arxiv.org/abs/1809.07454  
   Decision mapping: learned time-domain encoder/mask/decoder design and comparison context.
10. Subakan et al., “Attention is All You Need in Speech Separation,” arXiv:2010.13154.
    https://arxiv.org/abs/2010.13154  
    Decision mapping: SepFormer comparison context; this repository is not a reference
    SpeechBrain reproduction.
11. “SepReformer,” arXiv:2406.05983.
    https://arxiv.org/abs/2406.05983  
    Decision mapping: clamped PIT-SI-SDR training motivation used by the v2+ loss path.
12. LibriMix official repository: https://github.com/JorisCos/LibriMix  
    Decision mapping: dataset construction and 16-kHz Libri-based mixture conventions.
13. Asteroid official repository: https://github.com/asteroid-team/asteroid  
    Decision mapping: Conv-TasNet/DPRNN implementation and evaluation conventions.

14. SPIRAL official implementation: https://github.com/huawei-noah/Speech-Backbones/tree/main/SPIRAL  
   Decision mapping: clean/perturbed teacher-student representation matching and
   perturbation-invariant speech features.
15. RobustDistiller official repository: https://github.com/Hguimaraes/robustdistiller  
   Decision mapping: feature-denoising distillation and optional clean
   waveform/STFT reconstruction from noisy student features.
16. Teacher-student separation implementation:
   https://github.com/Sanyuan-Chen/CSS_with_TSTransformer  
   Decision mapping: layer-wise feature distillation, output distillation, and
   objective shifting for speech separation.
17. NASS official repository: https://github.com/TzuchengChang/NASS  
   Decision mapping: noise-aware source separation and patch-wise contrastive
   separation/noise representation learning.
18. Audio Barlow Twins: https://arxiv.org/abs/2209.14345  
   Decision mapping: redundancy reduction and anti-collapse representation
   invariance objective.

The repository also contains `research/bibliography.md`; it is the source of additional
context and links. Claims in this package are not presented as reproductions of any
reference result unless the experiment ledger explicitly says so.
