# Ultralytics YOLO 🚀, AGPL-3.0 license

from engine.results import Results

def load_results_from_txt(results_dir: str, frame_number: int) -> Results:
    """
    Loads results from .tx file for a single frame and creates Results object

    Args:
        results_dir (str): Path to directory with saved detection results. 
        frame_number (int): Frame number of results to load.

    Returns:
        Results containing results for the given frame
    """
    # TODO: implement this