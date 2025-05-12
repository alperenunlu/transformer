from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

import torch
from torch.utils.data import DataLoader


def get_dataloaders(
    vocab_size: int = 37_000,
    batch_size: int = 512,
    max_length: int = 256,
    num_examples: int = 5000,
):
    dsd = load_dataset("wmt14", "de-en")
    src, tgt = "en", "de"

    def pick(dsd, split, n):
        split = dsd[split] if n == -1 else dsd[split].select(range(n))
        return split.map(lambda x: x["translation"], remove_columns=["translation"])

    train_raw = pick(dsd, "train", num_examples)
    valid_raw = pick(dsd, "validation", num_examples // 10)
    test_raw = pick(dsd, "test", num_examples // 10)

    tokenizer = Tokenizer(BPE())

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<EOS>"],
        continuing_subword_prefix="",
    )

    def batch_iterator(batch_size=1000):
        for batch in train_raw.iter(batch_size=batch_size):
            yield batch["en"] + batch["de"]

    tokenizer.train_from_iterator(
        batch_iterator(), trainer=trainer, length=len(train_raw) * 2
    )

    tokenizer.post_processor = TemplateProcessing(
        single="$A <EOS>",
        special_tokens=[("<EOS>", tokenizer.token_to_id("<EOS>"))],
    )

    BOS_ID = tokenizer.token_to_id("<pad>")

    def tokenize_fn(examples):
        inputs = examples[src]
        targets = examples[tgt]
        inputs_enc = tokenizer.encode_batch(inputs)
        targets_enc = tokenizer.encode_batch(targets)
        for enc_src, enc_tgt in zip(inputs_enc, targets_enc):
            enc_src.pad(max_length)
            enc_src.truncate(max_length)
            enc_tgt.pad(max_length)
            enc_tgt.truncate(max_length)
        return {
            "src": [x.ids for x in inputs_enc],
            "src_mask": [x.attention_mask for x in inputs_enc],
            "tgt": [[BOS_ID] + x.ids[:-1] for x in targets_enc],
            "lbl": [x.ids for x in targets_enc],
        }

    train_tok = train_raw.map(tokenize_fn, batched=True, remove_columns=[src, tgt])
    valid_tok = valid_raw.map(tokenize_fn, batched=True, remove_columns=[src, tgt])
    test_tok = test_raw.map(tokenize_fn, batched=True, remove_columns=[src, tgt])

    def collate_fn(batch):
        src = torch.tensor([b["src"] for b in batch], dtype=torch.long)
        src_mask = torch.tensor([b["src_mask"] for b in batch], dtype=torch.float)
        tgt = torch.tensor([b["tgt"] for b in batch], dtype=torch.long)
        lbl = torch.tensor([b["lbl"] for b in batch], dtype=torch.long)
        return src, src_mask, tgt, lbl

    train_loader = DataLoader(
        train_tok, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_tok, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_tok, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader, tokenizer
