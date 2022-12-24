from torch.utils.tensorboard import SummaryWriter

writer = None  # TensorBoard SummaryWriter instance


def _log_scalars(scalars, step=0):
    for k, v in scalars.items():
        writer.add_scalar(k, v, step)


def on_train_start(trainer):
    global writer
    writer = SummaryWriter(str(trainer.save_dir))
    trainer.console.info(f"Logging results to {trainer.save_dir}\n"
                         f"Starting training for {trainer.args.epochs} epochs...")


def on_batch_end(trainer):
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)


def on_val_end(trainer):
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = {"on_train_start": on_train_start, "on_val_end": on_val_end, "on_batch_end": on_batch_end}
