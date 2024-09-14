# aiscreening/__init__.py
import sys
from . import main  # Exposes the main module as aiscreening.main
from . import onions  # Exposes the onions module as aiscreening.onions

# Optionally, declare what is available when `from aiscreening import *` is used
__all__ = ['main', 'onions']

