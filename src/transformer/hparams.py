from dataclasses import dataclass, field, asdict
from tomllib import load


@dataclass
class ModelHyperparameters:
    vocab_size: int = 37_000
    hidden_size: int = 512
    dropout: float = 0.0
    num_hidden_layers: int = 6
    filter_size: int = 2048
    num_heads: int = 8
    attention_key_channels: int = 0
    attention_value_channels: int = 0
    attention_dropout: float = 0.0
    relu_dropout: float = 0.0
    residual_dropout: float = 0.1


@dataclass
class TrainingHyperparameters:
    batch_size: int = 4096
    max_length: int = 256
    clip_grad_norm: float = 0.0
    optimizer_adam_epsilon: float = 1e-8
    learning_rate: float = 0.1
    learning_rate_warmup_steps: int = 4000
    weight_decay: float = 0.0
    optimizer_adam_beta1: float = 0.9
    optimizer_adam_beta2: float = 0.98
    label_smoothing: float = 0.1
    max_steps: int = 100_000


@dataclass
class Hyperparameters:
    model: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    def asdict(self):
        return asdict(self)


def load_hyperparameters_from_toml(path: str) -> Hyperparameters:
    with open(path, "rb") as f:
        data = load(f)
    return Hyperparameters(
        model=ModelHyperparameters(**data.get("model", {})),
        training=TrainingHyperparameters(**data.get("training", {})),
    )
