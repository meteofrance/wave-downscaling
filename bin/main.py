"""Main script to use the model with pytorch lightning CLI.
Train with fit and infer with predict.

Exemple usage:
    runai python bin/main.py fit --config bin/config_test_cli.yaml
"""

from lightning.pytorch.cli import LightningCLI
from ww3.datamodule import WW3DataModule
from ww3.plmodules.baseplmodule import WW3BaseLightningModule


class WW3CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.wave_params", "model.init_args.wave_params")


if __name__ == "__main__":
    WW3CLI(
        model_class=WW3BaseLightningModule,
        datamodule_class=WW3DataModule,
        subclass_mode_model=True,
    )
