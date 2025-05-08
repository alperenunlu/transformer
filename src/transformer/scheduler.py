from torch.optim.lr_scheduler import LRScheduler


class NoamLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 4000,
        d_model: int = 512,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lr = self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        return [lr] * len(self.base_lrs)
