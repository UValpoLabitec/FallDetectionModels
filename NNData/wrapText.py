import glob
from random import shuffle

read_files = glob.glob("*.txt")
shuffle(read_files)

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())