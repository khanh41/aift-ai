from pathlib import Path

ANGLE_TO_CALCULATE = (
    ((2, 1), (2, 3)),  # VAI TRAI
    ((5, 1), (5, 6)),  # VAI PHAI
    ((3, 2), (3, 4)),  # KHUYU TAY TRAI
    ((6, 5), (6, 7)),  # KHUYU TAY PHAI
    ((1, 2), (1, 8)),  # COT SONG VOI VAI
    # ((8,1),(8,9)), # HONG VOI COT SONG
    ((10, 9), (10, 11)),  # DAU GOI TRAI
    ((13, 12), (13, 14)),  # DAU GOI PHAI
)

FIREBASE_IMAGE_URL = lambda x: f"https://firebasestorage.googleapis.com/v0/b/aift-b7b2c.appspot.com/o/{x}.jpg?alt=media"

BACKEND_URL = "http://localhost:8088/api"

ROOT_PATH = str(Path(__file__).parent.parent)
