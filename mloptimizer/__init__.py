# Configure logging for the library
# Following Python best practices: library adds NullHandler, user configures output
import logging

logging.getLogger("mloptimizer").addHandler(logging.NullHandler())
