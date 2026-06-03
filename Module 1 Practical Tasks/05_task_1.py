import csv
import json
from pathlib import Path

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self, min_score=None):
        if not self.file_path.exists():
            print(f"Error: File not found at {self.file_path}")
            return []

        records = []
        suffix = self.file_path.suffix.lower()

        if suffix == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                records = json.load(f)

        elif suffix == ".jsonl":
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    cleaned_line = line.strip()
                    if cleaned_line:
                        records.append(json.loads(cleaned_line))

        elif suffix == ".csv":
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "score" in row:
                        row["score"] = float(row["score"])
                    records.append(row)
        else:
            print(f"Unsupported file format: {suffix}")
            return []

        if min_score is not None:
            filtered_records = []
            for item in records:
                if "score" in item and item["score"] >= min_score:
                    filtered_records.append(item)
            return filtered_records

        return records

if __name__ == "__main__":
    sample_csv = Path("test_data.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "score"])
        writer.writerow(["What is AI?", "Artificial Intelligence", "0.95"])
        writer.writerow(["What is 2+2?", "Four", "0.50"])

    loader = DatasetLoader("test_data.csv")
    
    print("all records:")
    print(loader.load())
    
    print("\nfiltered records(Score >= 0.80):")
    print(loader.load(min_score=0.80))
    sample_csv.unlink()