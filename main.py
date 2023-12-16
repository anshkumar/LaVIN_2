import torch
from lightning.pytorch.plugins import BitsandbytesPrecision # noqa: F401
from lavin import LightningTransformer # noqa: F401
from util.datasets import ScienceQADataModule # noqa: F401
from pytorch_lightning.utilities import cli as pl_cli

cli = pl_cli.LightningCLI()
