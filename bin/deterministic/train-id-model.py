import logging
import os
from pathlib import Path

from codetiming import Timer
import typer

app = typer.Typer()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger()
logger.setLevel("INFO")


@app.command()
@Timer(name="train", text="{name}: {minutes:.1f} minutes", logger=logging.info)
def main(
    workdir: Path,
):

    os.makedirs(workdir, exist_ok=True)

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    logging.info(f"Starting {os.path.basename(__file__)}")

    logging.info(f"Finished {os.path.basename(__file__)}")


if __name__ == "__main__":
    app()
