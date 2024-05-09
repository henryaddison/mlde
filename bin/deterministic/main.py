import ml_downscaling_emulator.deterministic.run_lib as run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

from knockknock import slack_sender

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train"], "Running mode: train.")
flags.mark_flags_as_required(["workdir", "config", "mode"])


@slack_sender(webhook_url=os.getenv("KK_SLACK_WH_URL"), channel="general")
def main(argv):
    if FLAGS.mode == "train":
        # Create the working directory
        os.makedirs(FLAGS.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
