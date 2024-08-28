import csv
import os


class DataExtractor:
    def __init__(self, playerName: str) -> None:
        self._playerName = playerName
        self._resetFile()

    def _resetFile(self) -> None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{self._playerName}.csv")
        with open(file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                ["last_play"])

    def saveData(self, data: str) -> None:
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{self._playerName}.csv")
        with open(file_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)
