import logging
import os


def init_logger(filename='mloptimizer.log', log_path=".", debug=False, file_output=True):
    """
    Initializes a logger and returns it.

    Parameters
    ----------
    filename : str, optional
        The name of the log file. The default is 'mloptimizer.log'.
    log_path : str, optional
        The path of the log file. The default is ".".
    debug : bool, optional
        Activate debug level. The default is False.
    file_output : bool, optional
        If True, logs to file. If False, only logs to console. The default is True.
    Returns
    -------
    custom_logger : logging.Logger
        The logger.
    logfile_path : str or None
        The path to the log file if file_output=True, otherwise None.
    """
    # Some logger variables
    logfile_path = os.path.join(log_path, filename) if file_output else None

    log_level = logging.INFO

    if debug:
        log_level = logging.DEBUG

    # Create a custom logger
    custom_logger = logging.getLogger(filename)
    custom_logger.setLevel(log_level)

    # Create logger formatter
    log_format = (
        "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d")
    logger_formatter = logging.Formatter(log_format)

    # Create file handler only if file output is enabled
    if file_output:
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logger_formatter)
        custom_logger.addHandler(file_handler)

    # Console handler - set to INFO level so messages are visible
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logger_formatter)
    custom_logger.addHandler(console_handler)

    # custom_logger.propagate = False

    # Logger configured
    custom_logger.debug("Logger configured")
    return custom_logger, logfile_path


def create_optimization_folder(folder):
    """
    Creates a folder to save the results of the optimization.

    Parameters
    ----------
    folder : str
        The path of the folder to create.

    Returns
    -------
    folder : str
        The path of the folder created.
    """
    if os.path.exists(folder):
        logging.warning("The folder {} already exists and it will be used".format(folder))
    elif os.makedirs(folder, exist_ok=True):
        logging.warning("The folder {} has been created.".format(folder))
    else:
        logging.error("The folder {} could not be created.".format(folder))
    return folder
