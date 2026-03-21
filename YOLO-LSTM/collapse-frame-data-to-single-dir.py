import os
from pathlib import Path
import shutil

def main():
    counter = 0
    os.makedirs("unlabeled-frame-data", exist_ok=True)
    p = Path("mario_dataset/human_play_images")
    out = "unlabeled-frame-data"
    for dir in p.iterdir():
        for f in os.scandir(dir):
            if f.is_file():
                tmp = os.path.abspath(dir)
                og_fname = os.path.join(tmp, f.name)
                new_fname = os.path.join(out, f"frame_{counter}.png")
                shutil.copy(og_fname, new_fname)
                counter+=1

main()