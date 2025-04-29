from olmo.config import TrainConfig
from olmo.config import (
    ActivationCheckpointingStrategy,
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
import torch
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.checkpoint import FullCheckpointer, TorchNewStyleShardedCheckpointer
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

# Load config
cfg = TrainConfig.load("/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base-0924/config.yaml")

# Force CPU initialization for checkpoint loading
cfg.model.init_device = "cpu"  # Critical fix

# Initialize components
base_model = OLMo(cfg.model)

# Load checkpoint
full_checkpointer = FullCheckpointer(cfg)
model_state, _ = full_checkpointer.load_checkpoint(
    load_path="/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/base-0924/",
    load_optimizer_state=False
)
base_model.load_state_dict(model_state)

# Configure FSDP with proper strategy mapping
sharding_map = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    "NO_SHARD": ShardingStrategy.NO_SHARD
}

fsdp_model = FSDP(
    base_model,
    device_id=torch.cuda.current_device(),
    sharding_strategy=cfg.fsdp.sharding_strategy,
    mixed_precision=cfg.fsdp_precision,  # Match your config's precision settings
)

# Save sharded checkpoint
sharded_checkpointer = build_sharded_checkpointer(cfg)
sharded_checkpointer.save_checkpoint(
    dir="/home/mila/k/khandela/scratch/ai2-llm/checkpoints/OLMoE/sharded-base-0924",
    dist_model=fsdp_model,
    optim=torch.optim.AdamW(fsdp_model.parameters()),  # Dummy optimizer
    trainer_state={}
)