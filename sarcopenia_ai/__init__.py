
__version__ = '0.1.0'

import os
import logging
from logging import NullHandler
from logging import config

# logging.getLogger(__name__).addHandler(logging.NullHandler())

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.cfg')
config.fileConfig(log_file_path, disable_existing_loggers=False)
