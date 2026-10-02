"""Allow ``python -m dashas_drop_sim`` from the GUI subprocess."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
