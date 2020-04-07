import typer

app = typer.Typer(name = "MLP Cli tool")


@app.command(name = 'ls')
def ls():
    typer.echo("Hello")


# @app.command(name = 'test')
# def test():
    # typer.echo(tf.config.list_physical_devices("GPU"))


def main():
    app()

if __name__ == "__main__":
    # app()
    # print(tf.config.list_physical_devices("GPU"))
    print('hello')
