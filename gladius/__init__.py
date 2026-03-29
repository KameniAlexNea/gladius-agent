from gladius.config import LAYOUT, SETTINGS, load_project_env

__all__ = [
    "LAYOUT",
    "SETTINGS",
    "load_project_env",
]

if __name__ == "__main__":
    from gladius.cli import main

    main()
