# Non-RoFormer MNN Validation Report

Date: 2026-05-28

This report records the macOS validation for the remaining local non-RoFormer
models. Audio DSP, chunking, and overlap-add stay outside MNN. MNN runs the
fixed-shape neural core with `batch_size=1` and `overlap_size=0`.

## Models

| Preset | Source model | Core boundary | Input tensors | Output tensors |
| --- | --- | --- | --- | --- |
| `scnet_similarity` | `models/model_scnet_ep_102_sdr_12.8941.ckpt` | SCNet post-STFT core | `stft_repr [1,4,2049,130]` | `stft_out [1,4,2049,130]` |
| `scnet_difference` | `models/model_scnet_ep_30_sdr_15.1291.ckpt` | SCNet post-STFT core | `stft_repr [1,4,2049,130]` | `stft_out [1,4,2049,130]` |
| `htdemucs_similarity` | `models/model_htdemucs_ep_21_sdr_13.6970.ckpt` | HTDemucs frequency/time core | `mag [1,4,2048,130]`, `mix [1,2,132300]` | `mask [1,2,4,2048,130]`, `time [1,2,2,132300]` |
| `smoke_scnet` | `models/smoke/scnet/scnet_checkpoint_musdb18.ckpt` | SCNet post-STFT core | `stft_repr [1,4,2049,476]` | `stft_out [1,16,2049,476]` |
| `smoke_htdemucs` | `models/smoke/htdemucs/HTDemucs4.th` | HTDemucs frequency/time core | `mag [1,4,2048,474]`, `mix [1,2,485100]` | `mask [1,4,4,2048,474]`, `time [1,4,2,485100]` |
| `smoke_mdx23c` | `models/smoke/mdx23c/model_vocals_mdx23c_sdr_10.17.ckpt` | MDX23C post-subband-STFT core | `cws_spec [1,16,1024,256]` | `cac_spec [1,8,4096,256]` |
| `smoke_apollo` | `models/smoke/apollo/Apollo_LQ_MP3_restoration.ckpt` | Apollo post-STFT core | `stft_ri [2,2,442,301]` | `est_spec_ri [2,2,442,301]` |
| `smoke_bandit` | `models/smoke/bandit/model_bandit_plus_dnr_sdr_11.47.chpt` | Bandit post-STFT core | `stft_ri [1,2,2,1025,517]` | `est_spec_ri [1,3,2,2,1025,517]` |
| `smoke_bandit_v2` | `models/smoke/bandit_v2/checkpoint-multi_state_dict.ckpt` | Bandit v2 post-STFT core | `stft_ri [2,1,2,1025,751]` | `est_spec_ri [2,3,1,2,1025,751]` |

SCNet full-graph export is not used because ONNX export fails on complex
`torch.stft`. The SCNet internal `rfft/irfft` feature conversions are exported
as fixed real-valued matrix multiplications for each preset's fixed time shape.

HTDemucs full-graph export is also avoided. The validated boundary keeps
`_spec`, `_mask`, and `_ispec` outside MNN while the encoder, cross-transformer,
decoder, and time branch run in MNN.

MDX23C keeps `SubbandSTFT` and inverse outside MNN. Apollo keeps STFT/ISTFT
outside MNN and exports the band feature extractor, Apollo network, and
spectrogram output heads. Bandit and Bandit v2 keep STFT/ISTFT outside MNN and
export real-valued band-split, sequence modelling, mask estimation, and complex
multiplication. Their export wrapper uses real/imag tensors throughout to avoid
ONNX complex ops.

`models/smoke/segm_models` and `models/smoke/swin_upernet` are not included in
this table because the current `pymss` package does not support
`model_type=segm_models` or `model_type=swin_upernet` in
`get_model_from_config` / `MSSeparator`. They only appear in old
`models/logic_bs_roformer` helper code.

## Artifacts

Artifacts were generated under ignored `benchmark_results/mnn_work/`.

| Preset | ONNX | MNN | Metadata |
| --- | ---: | ---: | --- |
| `scnet_similarity` | 41 MB | 41 MB | `benchmark_results/mnn_work/scnet_similarity/scnet_similarity_metadata.json` |
| `scnet_difference` | 41 MB | 41 MB | `benchmark_results/mnn_work/scnet_difference/scnet_difference_metadata.json` |
| `htdemucs_similarity` | 163 MB | 163 MB | `benchmark_results/mnn_work/htdemucs_similarity/htdemucs_similarity_metadata.json` |
| `smoke_scnet` | 42 MB | 46 MB | `benchmark_results/mnn_work/smoke_scnet/smoke_scnet_metadata.json` |
| `smoke_htdemucs` | 168 MB | 168 MB | `benchmark_results/mnn_work/smoke_htdemucs/smoke_htdemucs_metadata.json` |
| `smoke_mdx23c` | 427 MB | 428 MB | `benchmark_results/mnn_work/smoke_mdx23c/smoke_mdx23c_metadata.json` |
| `smoke_apollo` | 64 MB | 64 MB | `benchmark_results/mnn_work/smoke_apollo/smoke_apollo_metadata.json` |
| `smoke_bandit` | 188 MB | 237 MB | `benchmark_results/mnn_work/smoke_bandit/smoke_bandit_metadata.json` |
| `smoke_bandit_v2` | 207 MB | 279 MB | `benchmark_results/mnn_work/smoke_bandit_v2/smoke_bandit_v2_metadata.json` |

## Core Validation

Random fixed-shape validation compares PyTorch, ONNX Runtime, and MNN on CPU.

| Preset | Output | MNN vs PyTorch mean_abs | MNN vs PyTorch rmse | ref_rms |
| --- | --- | ---: | ---: | ---: |
| `scnet_similarity` | `stft_out` | `6.4495e-05` | `3.4414e-04` | `2.3328e-01` |
| `scnet_difference` | `stft_out` | `3.1383e-03` | `3.0509e-02` | `2.9768e+00` |
| `htdemucs_similarity` | `mask` | `2.3249e-05` | `3.5052e-05` | `3.8020e-02` |
| `htdemucs_similarity` | `time` | `6.7030e-05` | `1.0950e-04` | `6.9361e-01` |
| `smoke_scnet` | `stft_out` | `1.1545e-04` | `9.2640e-04` | `1.9120e+00` |
| `smoke_htdemucs` | `mask` | `1.3194e-04` | `3.2788e-04` | `3.0030e+00` |
| `smoke_htdemucs` | `time` | `1.0597e-05` | `2.3179e-05` | `4.8851e-01` |
| `smoke_mdx23c` | `cac_spec` | `4.5231e-05` | `1.5992e-04` | `1.0529e+00` |
| `smoke_apollo` | `est_spec_ri` | `8.5762e-07` | `5.4404e-06` | `4.5755e-01` |
| `smoke_bandit` | `est_spec_ri` | `2.3998e-06` | `5.5859e-06` | `4.9475e-01` |
| `smoke_bandit_v2` | `est_spec_ri` | `9.2442e-08` | `2.7189e-07` | `5.7690e-01` |

The SCNet `difference` random tensor case is noisier than real-audio validation,
so the end-to-end audio check is the practical quality gate for that model.

## Audio Validation

Three-second `test.m4a` validation used the lowest-memory settings:
`batch_size=1`, `overlap_size=0`, `threads=1`.

| Preset | Stem | mean_abs | rmse | MNN seconds | PyTorch seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `scnet_similarity` | `similarity` | `5.1015e-06` | `8.0629e-06` | `1.02` | `2.06` |
| `scnet_difference` | `difference` | `5.3106e-06` | `1.0000e-05` | `0.90` | `2.05` |
| `htdemucs_similarity` | `similarity` | `3.4557e-05` | `5.7889e-05` | `0.47` | `0.60` |
| `htdemucs_similarity` | `difference` | `3.2613e-05` | `5.4430e-05` | `0.47` | `0.60` |
| `smoke_scnet` | `drums` | `5.1873e-06` | `8.4502e-06` | `2.58` | `6.15` |
| `smoke_scnet` | `bass` | `1.4652e-06` | `2.4045e-06` | `2.58` | `6.15` |
| `smoke_scnet` | `other` | `9.9274e-06` | `1.6567e-05` | `2.58` | `6.15` |
| `smoke_scnet` | `vocals` | `3.6195e-06` | `6.1365e-06` | `2.58` | `6.15` |
| `smoke_htdemucs` | `drums` | `3.1674e-07` | `7.4594e-07` | `0.98` | `1.48` |
| `smoke_htdemucs` | `bass` | `1.3903e-07` | `3.7778e-07` | `0.98` | `1.48` |
| `smoke_htdemucs` | `other` | `1.7395e-06` | `3.6783e-06` | `0.98` | `1.48` |
| `smoke_htdemucs` | `vocals` | `9.1417e-08` | `2.0803e-07` | `0.98` | `1.48` |
| `smoke_mdx23c` | `vocals` | `1.0087e-05` | `1.7527e-05` | `0.84` | `0.95` |
| `smoke_mdx23c` | `other` | `1.0259e-05` | `1.7865e-05` | `0.84` | `0.95` |
| `smoke_apollo` | `restored` | `2.9168e-07` | `5.1991e-07` | `0.39` | `0.50` |
| `smoke_bandit` | `speech` | `2.5389e-11` | `3.4510e-11` | `39.27` | `6.03` |
| `smoke_bandit` | `music` | `1.2219e-08` | `1.7394e-08` | `39.27` | `6.03` |
| `smoke_bandit` | `effects` | `1.5267e-11` | `2.1432e-11` | `39.27` | `6.03` |
| `smoke_bandit_v2` | `speech` | `3.5414e-11` | `5.1317e-11` | `57.15` | `6.67` |
| `smoke_bandit_v2` | `music` | `1.0722e-08` | `1.5296e-08` | `57.15` | `6.67` |
| `smoke_bandit_v2` | `sfx` | `2.0794e-10` | `2.8199e-10` | `57.15` | `6.67` |

Output WAVs are under:

- `benchmark_results/mnn_work/scnet_similarity/audio_short`
- `benchmark_results/mnn_work/scnet_difference/audio_short`
- `benchmark_results/mnn_work/htdemucs_similarity/audio_short`
- `benchmark_results/mnn_work/smoke_scnet/audio_short`
- `benchmark_results/mnn_work/smoke_htdemucs/audio_short`
- `benchmark_results/mnn_work/smoke_mdx23c/audio_short`
- `benchmark_results/mnn_work/smoke_apollo/audio_short`
- `benchmark_results/mnn_work/smoke_bandit/audio_short`
- `benchmark_results/mnn_work/smoke_bandit_v2/audio_short`

## Commands

```sh
python tools/mnn_export/export_non_roformer_core.py --preset scnet_similarity --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset scnet_difference --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset htdemucs_similarity --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset smoke_scnet --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset smoke_htdemucs --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset smoke_mdx23c --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset smoke_apollo --out-dir benchmark_results/mnn_work
python tools/mnn_export/export_non_roformer_core.py --preset smoke_bandit --out-dir benchmark_results/mnn_work --convert-timeout 1800
python tools/mnn_export/export_non_roformer_core.py --preset smoke_bandit_v2 --out-dir benchmark_results/mnn_work --convert-timeout 1800

python tools/mnn_export/validate_non_roformer_core.py --preset scnet_similarity --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset scnet_difference --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset htdemucs_similarity --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_scnet --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_htdemucs --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_mdx23c --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_apollo --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_bandit --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_non_roformer_core.py --preset smoke_bandit_v2 --out-dir benchmark_results/mnn_work --threads 1

python tools/mnn_export/separate_non_roformer_with_mnn.py --preset scnet_similarity --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/scnet_similarity/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset scnet_difference --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/scnet_difference/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset htdemucs_similarity --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/htdemucs_similarity/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_scnet --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_scnet/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_htdemucs --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_htdemucs/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_mdx23c --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_mdx23c/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_apollo --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_apollo/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_bandit --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_bandit/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_non_roformer_with_mnn.py --preset smoke_bandit_v2 --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/smoke_bandit_v2/audio_short --seconds 3 --threads 1 --compare-torch
```

## Runtime Status

These validations use Python/PyMNN runners. The shared C++ runtime remains
torch-free for the RoFormer path; C++ host implementations for SCNet, HTDemucs,
MDX23C, Apollo, and Bandit DSP/core wiring are still future work.
