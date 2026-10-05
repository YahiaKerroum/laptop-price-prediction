"""Allow ``python -m laptop_price <command>``."""

from laptop_price.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
