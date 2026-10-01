"""Allow ``python -m dashas_drop_sim`` from the GUI subprocess."""

from .cli import main


raise SystemExit(main())
