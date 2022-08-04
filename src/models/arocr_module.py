from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import WordErrorRate
from torchmetrics import CharErrorRate


class ArocrLitModule(LightningModule):
    """Example of LightningModule 

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_cer = CharErrorRate()
        self.val_cer = CharErrorRate()
        self.test_cer = CharErrorRate()
        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()
    
        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        print(batch)
        x, y = batch
        print(x.shape)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        cer = self.train_cer(preds, targets)
        wer = self.train_wer(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/cer", cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/wer", wer, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_cer.reset()
        self.train_wer.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        cer = self.val_cer(preds, targets)
        wer = self.val_wer(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/cer", cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", wer, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_cer.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/cer_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_cer.reset()
        self.val_wer.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        cer = self.test_cer(preds, targets)
        wer = self.test_wer(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/cer", cer, on_step=False, on_epoch=True)
        self.log("test/wer", wer, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_cer.reset()
        self.test_wer.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "arocr.yaml")
    _ = hydra.utils.instantiate(cfg)
