import typer

APP_NAME = "MLP CLI tool"

app = typer.Typer(name=APP_NAME)


@app.command(name="ls")
def ls():
    configPath = Path(__file__).absolute()
    typer.echo(configPath)


@app.command(name="hello")
def hello():
    typer.echo("Hello")


def main():
    app()


if __name__ == "__main__":
    from pathlib import Path

    app()
