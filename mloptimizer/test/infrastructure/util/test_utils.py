import os
from mloptimizer.infrastructure.util.utils import create_optimization_folder


def test_create_optimization_folder(tmp_path):
    # Create a temporary path for testing
    folder_path = os.path.join(tmp_path, 'opt_folder')

    # Test creating a new folder
    returned_path = create_optimization_folder(folder_path)
    assert os.path.isdir(folder_path), "Folder should be created"
    assert returned_path == folder_path, "Returned path should match the created folder path"

    # Test creating an existing folder
    returned_path = create_optimization_folder(folder_path)
    assert returned_path == folder_path, "Returned path should match the existing folder path"
