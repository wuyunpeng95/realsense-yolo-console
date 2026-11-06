"""Download only the official, known YOLO11n weights into this project."""
from pathlib import Path
from urllib.request import urlretrieve

directory = Path(__file__).resolve().parent / "models"
directory.mkdir(exist_ok=True)
target = directory / "yolo11n.pt"
if target.is_file():
    print(f"Model already exists: {target}")
else:
    temporary = target.with_suffix(".download")
    try:
        urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded: {target}")
