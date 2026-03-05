import subprocess
import sys
import os

print(os.path.abspath(os.curdir))
print(os.environ["PATH"])
command = [
    sys.executable,
    "-m",
    "montreal_forced_aligner",
    "align",
    "corpus",
    "english_us_arpa",
    "english_mfa",
    "aligned"
]

subprocess.run(command, check=True)
