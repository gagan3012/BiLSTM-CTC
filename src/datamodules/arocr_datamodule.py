from typing import Any, Dict, Optional, Tuple
from transformers import TrOCRProcessor
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import datasets
from torchvision.transforms import transforms
import pandas as pd
import numpy as np

def preprocess(examples, processor, max_target_length=128):
    text = examples["text"]
    image = examples["image"].convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        text, padding="max_length", max_length=max_target_length
    ).input_ids
    labels = [label if label != processor.tokenizer.pad_token_id else -100 for label in labels]
    encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
    return encoding



class ArocrDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "OnlineKhatt",
        cache_dir: str = "data/cache/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        # datasets.load_dataset(self.hparams.data_dir,self.hparams.dataset_name,cache_dir=self.hparams.cache_dir)
        datasets.load_dataset("gagan3012/OnlineKhatt")
        TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # load dataset
            # dataset = datasets.load_dataset(self.hparams.data_dir,self.hparams.dataset_name,cache_dir=self.hparams.cache_dir)
            dataset = datasets.load_dataset("gagan3012/OnlineKhatt")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            fn_kwargs = dict(
                processor = processor,
            )
            dataset = dataset.map(preprocess, fn_kwargs=fn_kwargs)
            # split dataset
            self.data_train = dataset['train']
            # self.data_val = pd.DataFrame(dataset['validation'])
            self.data_val = dataset['dev']
            self.data_test = dataset['test']

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x: x,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "arocr.yaml")
    cfg.data_dir = str(root / "data")
    cfg.cache_dir = str(root / "data" / "cache")
    dm = hydra.utils.instantiate(cfg)
    dm.prepare_data()
    dm.setup('fit')
    print(next(iter(dm.train_dataloader())))