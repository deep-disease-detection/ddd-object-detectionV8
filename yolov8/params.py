import os

YOLO_DATA_PATH = os.environ.get("YOLO_DATA_PATH")
YOLO_IMAGE_SIZE = 1024
YOLO_IMAGE_PATH = os.environ.get("YOLO_IMAGE_PATH")
YOLO_LABELS_PATH = os.environ.get("YOLO_LABELS_PATH")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")


VIRUSES = [
    "Adenovirus",
    "Astrovirus",
    "CCHF",
    "Cowpox",
    "Ebola",
    "Marburg",
    "Norovirus",
    "Orf",
    "Papilloma",
    "Rift Valley",
    "Rotavirus",
]

VIRUS_METADATA = {
    "Adenovirus": {"id": 0, "elongated": False, "diameter": 80},
    "Astrovirus": {"id": 1, "elongated": False, "diameter": 25},
    "CCHF": {"id": 2, "elongated": False, "diameter": 150},
    "Cowpox": {"id": 3, "elongated": False, "diameter": 350},
    "Ebola": {"id": 4, "elongated": True, "diameter": 80},
    "Marburg": {"id": 5, "elongated": True, "diameter": 80},
    "Norovirus": {"id": 6, "elongated": False, "diameter": 30},
    "Orf": {"id": 7, "elongated": False, "diameter": 320},
    "Papilloma": {"id": 8, "elongated": False, "diameter": 55},
    "Rift Valley": {"id": 9, "elongated": False, "diameter": 90},
    "Rotavirus": {"id": 10, "elongated": False, "diameter": 80},
}
