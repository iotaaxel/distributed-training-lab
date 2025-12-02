"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, rank: Optional[int] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        rank: Process rank for distributed training (only rank 0 logs by default)
    """
    if rank is not None and rank != 0:
        # Only log from rank 0 in distributed settings
        logging.basicConfig(level=logging.WARNING)
        return
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

