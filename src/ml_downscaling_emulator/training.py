from contextlib import contextmanager
import logging


def log_epoch(epoch, epoch_metrics, tb_writer):
    import mlflow

    logging.info(
        " ".join(
            [f"epoch {epoch},"] + [f"{k}: {v:.5e}" for k, v in epoch_metrics.items()]
        )
    )

    mlflow.log_metrics(epoch_metrics, step=epoch)
    for name, value in epoch_metrics.items():
        tb_writer.add_scalar(name, value, epoch)


@contextmanager
def track_run(experiment_name, run_name, config, tags, tb_dir):
    import mlflow
    from torch.utils.tensorboard import SummaryWriter

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({key: True for key in tags})
        mlflow.log_params(config)

        with SummaryWriter(tb_dir) as tb_writer:
            yield tb_writer
