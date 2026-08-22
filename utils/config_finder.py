from pathlib import Path

def find_config_path(start_file:str, file_name: str = "config.yaml", marker_dir: str = "configs") -> Path:
    """
    Function used to return the path of the config.yaml file

    Params:
        start_file (str): file path of file which calls this function
        file_name (str): name of file whose path we are looking for
        marker_dir (str): directory in which required file is found

    Returns:
        The file path where the required file is found
    """

    current_dir = Path(start_file).resolve().parent

    for _ in range(4):
        candidate = current_dir / marker_dir / file_name

        if candidate.is_file():
            return candidate
        
        if current_dir.parent == current_dir:
            break

        current_dir = current_dir.parent

    raise FileNotFoundError(
        f"Could not find {marker_dir}/{file_name} by walking up from {start_file}"
    )