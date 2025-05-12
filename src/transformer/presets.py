from transformer.hparams import Hyperparameters


def transformer_base():
    return Hyperparameters()


def transformer_single_gpu():
    hparams = transformer_base()
    hparams.training.batch_size = 8192
    hparams.training.learning_rate_warmup_steps = 16_000
    return hparams


def transformer_tiny():
    hparams = transformer_base()
    hparams.model.hidden_size = 64
    hparams.model.filter_size = 128
    hparams.model.num_heads = 4
    return hparams


def transformer_l2():
    hparams = transformer_base()
    hparams.model.num_hidden_layers = 2
    return hparams


def transformer_l4():
    hparams = transformer_base()
    hparams.model.num_hidden_layers = 4
    return hparams


def transformer_l8():
    hparams = transformer_base()
    hparams.model.num_hidden_layers = 8
    return hparams


def transformer_h1():
    hparams = transformer_base()
    hparams.model.num_heads = 1
    return hparams


def transformer_h4():
    hparams = transformer_base()
    hparams.model.num_heads = 4
    return hparams


def transformer_h16():
    hparams = transformer_base()
    hparams.model.num_heads = 16
    return hparams


def transformer_h32():
    hparams = transformer_base()
    hparams.model.num_heads = 32
    return hparams


def transformer_k128():
    hparams = transformer_base()
    hparams.model.attention_key_channels = 128
    return hparams


def transformer_k256():
    hparams = transformer_base()
    hparams.model.attention_key_channels = 256
    return hparams


def transformer_ff1024():
    hparams = transformer_base()
    hparams.model.filter_size = 1024
    return hparams


def transformer_ff4096():
    hparams = transformer_base()
    hparams.model.filter_size = 4096
    return hparams


def transformer_dr0():
    hparams = transformer_base()
    hparams.model.residual_dropout = 0.0
    return hparams


def transformer_dr2():
    hparams = transformer_base()
    hparams.model.residual_dropout = 0.2
    return hparams


def transformer_ls0():
    hparams = transformer_base()
    hparams.training.label_smoothing = 0.0
    return hparams


def transformer_ls2():
    hparams = transformer_base()
    hparams.training.label_smoothing = 0.2
    return hparams


def transformer_hs256():
    hparams = transformer_base()
    hparams.model.hidden_size = 256
    return hparams


def transformer_hs1024():
    hparams = transformer_base()
    hparams.model.hidden_size = 1024
    return hparams


def transformer_big_dr1():
    hparams = transformer_base()
    hparams.model.hidden_size = 1024
    hparams.model.filter_size = 4096
    hparams.model.num_heads = 16
    hparams.model.residual_dropout = 0.1
    return hparams


def transformer_big_dr2():
    hparams = transformer_big_dr1()
    hparams.model.residual_dropout = 0.2
    return hparams


def transformer_big_dr3():
    hparams = transformer_big_dr1()
    hparams.model.residual_dropout = 0.3
    return hparams


def transformer_big_single_gpu():
    hparams = transformer_big_dr1()
    hparams.training.learning_rate_warmup_steps = 16000
    hparams.training.optimizer_adam_beta2 = 0.998
    return hparams


def transformer_parsing_base_dr6():
    hparams = transformer_base()
    hparams.model.attention_dropout = 0.2
    hparams.model.residual_dropout = 0.2
    hparams.training.max_length = 512
    hparams.training.learning_rate_warmup_steps = 16000
    hparams.model.hidden_size = 1024
    hparams.training.learning_rate = 0.5
    return hparams


def transformer_parsing_big():
    hparams = transformer_big_dr1()
    hparams.training.max_length = 512
    hparams.training.learning_rate_warmup_steps = 4000
    hparams.training.batch_size = 2048
    hparams.training.learning_rate = 0.5
    return hparams


PRESET_CONFIGS = {
    "base": transformer_base,
    "single_gpu": transformer_single_gpu,
    "tiny": transformer_tiny,
    "l2": transformer_l2,
    "l4": transformer_l4,
    "l8": transformer_l8,
    "h1": transformer_h1,
    "h4": transformer_h4,
    "h16": transformer_h16,
    "h32": transformer_h32,
    "k128": transformer_k128,
    "k256": transformer_k256,
    "ff1024": transformer_ff1024,
    "ff4096": transformer_ff4096,
    "dr0": transformer_dr0,
    "dr2": transformer_dr2,
    "ls0": transformer_ls0,
    "ls2": transformer_ls2,
    "hs256": transformer_hs256,
    "hs1024": transformer_hs1024,
    "big_dr1": transformer_big_dr1,
    "big_dr2": transformer_big_dr2,
    "big_dr3": transformer_big_dr3,
    "big_single_gpu": transformer_big_single_gpu,
    "parsing_base_dr6": transformer_parsing_base_dr6,
    "parsing_big": transformer_parsing_big,
}


def get_transformer_by_name(name="base"):
    try:
        return PRESET_CONFIGS[name]()
    except KeyError:
        raise ValueError(f"Unknown transformer configuration: '{name}'")
