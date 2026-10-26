from pytorch_lightning.callbacks import Callback

class LearningRateLogger(Callback):
    def on_epoch_end(self, trainer, pl_module):
        # 获取当前的学习率
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        print(
            f"Epoch {trainer.current_epoch}: Learning Rate = {current_lr:.8f}")
