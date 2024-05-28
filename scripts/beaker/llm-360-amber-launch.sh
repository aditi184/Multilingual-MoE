#!/usr/bin/env bash

set -ex

NUM_NODES=1

gantry run \
  --workspace ai2/OLMo-training  \
  --task-name llm-360-amber-data-repro \
  --description "Reproduction of LLM 360 Amber 7B training run" \
  --priority high \
  --beaker-image shanea/olmo-torch2.2-gantry \
  --cluster ai2/jupiter-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --synchronized-start-timeout 20m \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=DUSTINS_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/llm-360-amber.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK"
