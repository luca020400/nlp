import csv
import hashlib
import json


def read_json_config(path):
    """Load settings from a UTF-8 JSON file."""
    with open(path, encoding="utf-8") as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    return settings


def export_csv_rows(path, fieldnames, rows):
    """Write records to a CSV file with a header row."""
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_file_digest(path):
    """Calculate a SHA-256 checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
