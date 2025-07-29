"""
Command-line interface for photonic-nn-foundry.
"""

import click
import logging
from .core import PhotonicAccelerator
from .transpiler import torch2verilog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Photonic Neural Network Foundry CLI."""
    pass


@main.command()
@click.option("--pdk", default="skywater130", help="Process design kit")
@click.option("--wavelength", default=1550.0, help="Operating wavelength (nm)")
def init(pdk: str, wavelength: float):
    """Initialize a new photonic accelerator project."""
    accelerator = PhotonicAccelerator(pdk=pdk, wavelength=wavelength)
    click.echo(f"Initialized photonic accelerator with {pdk} PDK at {wavelength}nm")


@main.command()
@click.argument("model_path")
@click.option("--target", default="photonic_mac", help="Target architecture")
@click.option("--output", "-o", help="Output Verilog file")
def transpile(model_path: str, target: str, output: str):
    """Transpile PyTorch model to photonic Verilog."""
    click.echo(f"Transpiling {model_path} to {target} architecture")
    # Placeholder: would load and convert actual model
    if output:
        click.echo(f"Output written to {output}")


if __name__ == "__main__":
    main()