# from sys import path
from pathlib import Path
from pandas import read_csv, DataFrame as PandasDataFrame

def get_data_split(
    split: str = "train",
    base_dir: str = "data/binary_dataset"
) -> PandasDataFrame:
    """
    Load a dataset (train, test, or other split) from a CSV file.

    Parameters
    ----------
    split : str, default="train"
        Which dataset split to load (e.g., "train", "test", "valid").
    base_dir : str, default="data/binary_dataset"
        Base directory containing the CSV files.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    root_path = Path().resolve().parent
    dataset_path = root_path.joinpath(base_dir, f"{split}.csv")
    return load_dataset(dataset_path)

def get_dataset_path(data_dir: str = "data") -> Path:
    """
    Go one level up from the current working directory,
    then return the absolute path of the only CSV file inside `data_dir`.

    Parameters
    ----------
    data_dir : str, default="data"
        Directory containing the dataset.

    Returns
    -------
    Path
        Path to the single CSV file found.

    Raises
    ------
    FileNotFoundError
        If no CSV file is found in the directory.
    ValueError
        If more than one CSV file is found.
    """
    root_path = Path().resolve().parent
    data_path = root_path / data_dir
    csv_files = list(data_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found in {data_path}")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in {data_path}, expected only one.")

    # ici on sait qu’il y a exactement 1 fichier
    return csv_files[0]

def load_dataset(dataset_path: Path) -> PandasDataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.
    """
    return read_csv(dataset_path)
