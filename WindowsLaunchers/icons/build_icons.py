"""Build the multi-resolution Windows shortcut icon from its PNG source."""

from pathlib import Path

from PIL import Image


def main() -> None:
    directory = Path(__file__).resolve().parent
    with Image.open(directory / "drop-sim.png") as source:
        image = source.convert("RGBA")
        if image.width != image.height or image.getchannel("A").getextrema() != (0, 255):
            raise ValueError("Icon source must be square with transparent and opaque pixels")
        image.save(
            directory / "drop-sim.ico",
            format="ICO",
            sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
            bitmap_format="png",
        )


if __name__ == "__main__":
    main()
