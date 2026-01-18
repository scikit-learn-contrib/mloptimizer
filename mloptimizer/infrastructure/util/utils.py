import logging
import os

logger = logging.getLogger(__name__)


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
        logger.warning("The folder {} already exists and it will be used".format(folder))
    elif os.makedirs(folder, exist_ok=True):
        logger.warning("The folder {} has been created.".format(folder))
    else:
        logger.error("The folder {} could not be created.".format(folder))
    return folder
