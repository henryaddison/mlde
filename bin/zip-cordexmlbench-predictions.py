import dotenv
import logging
import os
from pathlib import Path
import typer
import zipfile

dotenv.load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)


def zip_submission(output_base: Path, emulator_id: str):
    # ZIP the submission using the "emulator_id" code specified in the registration
    zip_filename = f"{emulator_id}.zip"

    zip_path = os.path.join(output_base, zip_filename)

    logger.info(f"Creating submission package: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_base):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_base)
                zipf.write(abs_path, rel_path)


app = typer.Typer()


@app.command()
def main(
    path: Path,
    emulator_id: str = "RCMGEM-mv-orog",
):
    zip_submission(path, emulator_id)
    logger.info(f"DONE")


if __name__ == "__main__":
    app()
