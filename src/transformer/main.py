import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from argparse import ArgumentParser

import accelerate
from torchmetrics.text import Perplexity, BLEUScore

from transformer.config_loader import get_hparams
from transformer.model import Transformer
from transformer.scheduler import NoamLR
from transformer.data import get_dataloaders


class TransformerTrainer:
    def __init__(self, config, args, accelerator):
        self.config = config
        self.args = args
        self.accelerator = accelerator

        self.model = Transformer(config.model)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(
                config.training.optimizer_adam_beta1,
                config.training.optimizer_adam_beta2,
            ),
            eps=config.training.optimizer_adam_epsilon,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = NoamLR(
            optimizer=self.optimizer,
            warmup_steps=config.training.learning_rate_warmup_steps,
            d_model=config.model.hidden_size,
        )

        self.train_loader, self.valid_loader, self.test_loader, self.tokenizer = (
            get_dataloaders(
                vocab_size=config.model.vocab_size,
                batch_size=config.training.batch_size,
                max_length=config.training.max_length,
            )
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.token_to_id("<pad>"),
            label_smoothing=config.training.label_smoothing,
        )

        self.ppl = Perplexity().to(accelerator.device)
        self.bleu = BLEUScore().to(accelerator.device)

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.valid_loader,
            self.test_loader,
        ) = accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.valid_loader,
            self.test_loader,
        )

        self.step = 0
        self.checkpoint_interval = int(args.checkpointing_steps.replace("_", ""))

    def train_step(self):
        self.model.train()
        total_loss = 0
        for src, src_mask, tgt, lbl in tqdm(
            self.train_loader, desc="Training", leave=False
        ):
            pred = self.model(src, src_mask, tgt)
            loss = self.criterion(pred.view(-1, pred.size(-1)), lbl.view(-1))

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            self.step += 1
            total_loss += loss.item()

            self.accelerator.log(
                {
                    "train/loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "step": self.step,
                },
                step=self.step,
            )

            if self.step % self.checkpoint_interval == 0:
                self.save_checkpoint()

            if self.step >= self.config.training.max_steps:
                break

        return total_loss / len(self.train_loader)

    def evaluate(self, data_loader, split="valid"):
        self.model.eval()
        total_bleu, total_ppl = 0.0, 0.0

        with torch.no_grad():
            for src, src_mask, tgt, lbl in tqdm(
                data_loader, desc=f"{split.capitalize()} Eval", leave=False
            ):
                gen_pred, _ = self.model.beam_search(src, src_mask)

                predictions = self.tokenizer.decode_batch(gen_pred.tolist())
                references = self.tokenizer.decode_batch(tgt.tolist())

                bleu_score = self.bleu(predictions, references)
                total_bleu += bleu_score.item()

                logits = self.model(src, src_mask, tgt)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                ppl_score = self.ppl(log_probs, lbl)
                total_ppl += ppl_score.item()

        avg_bleu = total_bleu / len(data_loader)
        avg_ppl = total_ppl / len(data_loader)

        self.accelerator.log(
            {
                f"{split}/bleu": avg_bleu,
                f"{split}/ppl": avg_ppl,
            },
            step=self.step,
        )

        return avg_bleu, avg_ppl

    def save_checkpoint(self):
        self.accelerator.save_state(self.args.output_dir)

    def run(self):
        pbar_step = tqdm(total=self.config.training.max_steps, desc="Steps")

        while self.step < self.config.training.max_steps:
            avg_loss = self.train_step()
            bleu, ppl = self.evaluate(self.valid_loader, split="valid")

            pbar_step.set_postfix(loss=avg_loss, bleu=bleu, ppl=ppl)
            pbar_step.update(1)

        self.evaluate(self.test_loader, split="test")


if __name__ == "__main__":
    parser = ArgumentParser(description="Transformer training")
    parser.add_argument(
        "--config", type=str, default="tiny", help="Config preset or file path."
    )
    parser.add_argument(
        "--project_dir", type=str, default="logs", help="Experiment log directory."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="10_000",
        help="Steps between checkpoints.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Checkpoint directory."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision mode."
    )

    args = parser.parse_args()
    config = get_hparams(args.config)

    accelerator = accelerate.Accelerator(
        project_dir=args.project_dir,
        log_with="wandb",
        dynamo_backend="inductor" if torch.cuda.is_available() else None,
        mixed_precision=args.mixed_precision,
    )
    accelerator.init_trackers(project_name="Transformer", config=config.asdict())

    trainer = TransformerTrainer(config, args, accelerator)
    trainer.run()
