import typer

from . import postprocess, sample

app = typer.Typer()
app.add_typer(sample.app, name="sample")
app.add_typer(postprocess.app, name="postprocess")


if __name__ == "__main__":
    app()
