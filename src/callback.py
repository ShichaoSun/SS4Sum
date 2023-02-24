#!/usr/bin/env python
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from utils import save_json


class LoggingCallback(Callback):
    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        pl_module.logger.log_metrics({"learning_rate": pl_module.trainer.optimizers[0].param_groups[0]["lr"]})

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        print('Testing ends')
        print('--------------------------------------------')
        save_json(pl_module.metrics, pl_module.metrics_save_path)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)

    @rank_zero_only
    def on_init_start(self, trainer):
        print('Starting to init trainer!')
        print('--------------------------------------------')

    @rank_zero_only
    def on_init_end(self, trainer):
        print('trainer is init now')
        print('--------------------------------------------')

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        print('Training ends')
        print('--------------------------------------------')

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        print('Training starts')
        print('--------------------------------------------')

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        print('Testing starts')
        print('--------------------------------------------')