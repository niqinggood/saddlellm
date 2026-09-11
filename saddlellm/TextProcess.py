"""Compatibility import for :mod:`saddlellm.data.text_processing`.

New code should import from :mod:`saddlellm.data`.
"""

from .data.text_processing import (
    DataSourceHandler,
    TextCleaner,
    TextProcessor,
    load_config,
    main,
)

__all__ = [
    "DataSourceHandler",
    "TextCleaner",
    "TextProcessor",
    "load_config",
    "main",
]


if __name__ == "__main__":
    main()
