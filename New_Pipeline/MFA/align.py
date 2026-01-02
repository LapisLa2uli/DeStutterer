import subprocess
import os

# Paths (edit these)
CORPUS_DIR = "corpus"
OUTPUT_DIR = "aligned"
ACOUSTIC_MODEL = "english"
DICTIONARY = "english"

os.makedirs(OUTPUT_DIR, exist_ok=True)

command = [
    "mfa",
    "align",
    CORPUS_DIR,
    DICTIONARY,
    ACOUSTIC_MODEL,
    OUTPUT_DIR,
    "--clean",
    "--verbose"
]

subprocess.run(command, check=True)

print("Alignment complete.")


from praatio import textgrid

tg = textgrid.openTextgrid(
    "aligned/speaker1/sample.TextGrid",
    includeEmptyIntervals=False
)

word_tier = tg.getTier("words")

for entry in word_tier.entries:
    start, end, word = entry
    print(f"{word}: {start:.3f}s → {end:.3f}s")