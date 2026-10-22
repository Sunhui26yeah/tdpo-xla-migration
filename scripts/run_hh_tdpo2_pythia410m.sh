#!/bin/bash
# TDPO2 on Anthropic HH, Pythia-410M (Table 1 recipe of the paper, README step 3 command, scaled to 410M).
# Usage: run_hh_tdpo2_pythia410m.sh SFT_POLICY_PT EXP_NAME [extra hydra overrides, e.g. smoke caps]
#   smoke caps: n_examples=12864 eval_every=12800
SFT=$1; EXP=$2; shift 2
python -u "$(dirname "$0")/../train.py" model=pythia28 model.name_or_path=EleutherAI/pythia-410m \
  model.archive=$SFT datasets=[hh] loss=tdpo loss.alpha=0.5 loss.beta=0.1 loss.if_tdpo2=true \
  exp_name=$EXP gradient_accumulation_steps=16 batch_size=64 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 wandb.enabled=false seed=0 "$@"
