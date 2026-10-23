# Running TDPO on a TPU (PyTorch/XLA)

This fork runs the TDPO trainer on a single TPU chip with PyTorch/XLA (tested: TPU v5e-1, torch 2.9.0 + torch_xla 2.9.0).
The XLA path is enabled automatically when torch_xla is installed and no CUDA device is present.

    bash scripts/run_hh_tdpo2_pythia410m.sh /path/to/sft/policy.pt EXP_NAME

On XLA:
- `FSDPTrainer` runs as a single-device trainer (FSDP over one rank is plain single-device training).
- A float16 reference model is loaded in float32 with matmul precision "high" (the TPU has no float16).
- Chosen/rejected and prompt tensors are padded to 128-token buckets so only a few graphs are compiled.
- The torch_xla metrics report is written to the run directory at steps 50 and 100 and at the end.

Environment pins needed with the original requirements on current pip: `pyarrow==12.0.1 fsspec==2023.5.0 peft==0.3.0`.
