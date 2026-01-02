import json
from phonemizer import phonemize
#from aeneas.executetask import ExecuteTask
#from aeneas.task import Task
import tempfile
import importlib.util
import sys
import os


from typing import List, Dict

def find_stuttered_phonemes(stutters: List[Dict], phonemes: List[Dict]):
    """
    Matches YOLO-Stutter detections with phoneme alignment results.
    Returns a list of stutter events annotated with phonemes.
    """
    stuttered = []

    for s in stutters:
        s_start, s_end = s["start_time"], s["end_time"]
        matched_phonemes = []

        for p in phonemes:
            p_start, p_end = p["start"], p["end"]

            # Check if there's overlap
            if p_end > s_start and p_start < s_end:
                matched_phonemes.append(p["phoneme"])

        # Choose the most central phoneme (optional)
        if matched_phonemes:
            # For simplicity, pick the phoneme with largest overlap
            overlaps = []
            for p in phonemes:
                overlap = max(0, min(p["end"], s_end) - max(p["start"], s_start))
                overlaps.append((overlap, p["phoneme"]))
            phoneme = max(overlaps)[1]
        else:
            phoneme = None

        stuttered.append({
            "stutter_type": s.get("class", "unknown"),
            "phoneme": phoneme,
            "confidence": s.get("confidence", None),
            "start_time": s_start,
            "end_time": s_end,
        })

    return stuttered


def phoneme_align(audio_path, transcript, lang="en-us"):
    """
    Aligns phonemes to the given audio using aeneas + phonemizer.
    Returns a list of dicts: [{phoneme, start, end}, ...]
    """

    # Step 1: Generate phoneme sequence from text
    phoneme_seq = phonemize(transcript, language=lang.split("-")[0], backend="espeak", strip=True)
    phoneme_seq = phoneme_seq.replace(" ", "_")  # prevent splitting words

    # Save temporary transcript file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
        tf.write(phoneme_seq)
        text_path = tf.name

    # Step 2: Setup Aeneas alignment task
    # r=task language=eng|is_text_type=plain|os_task_file_format=json
    config_string = "task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config_string)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(text_path)

    output_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    task.sync_map_file_path_absolute = output_json

    # Step 3: Run the forced alignment
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    # Step 4: Parse JSON output
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    phonemes = []
    for fragment in data["fragments"]:
        if "lines" in fragment and len(fragment["lines"]) > 0:
            phon = fragment["lines"][0]
            start = float(fragment["begin"])
            end = float(fragment["end"])
            phonemes.append({
                "phoneme": phon,
                "start": start,
                "end": end
            })

    # Cleanup temp files
    os.remove(text_path)
    os.remove(output_json)

    return phonemes



module_path = os.path.abspath("D:\\stuff\\StutterProject\\New_Pipeline\\YOLOStutter\\yolo-stutter\\etc\\inference.py")
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# Add the *parent folder of yolo-stutter* to sys.path
parent_dir = os.path.abspath("D:\\stuff\\StutterProject\\New_Pipeline\\YOLOStutter\\yolo-stutter")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
spec = importlib.util.spec_from_file_location("inference", module_path)
inference = importlib.util.module_from_spec(spec)
sys.modules["inference"] = inference
spec.loader.exec_module(inference)
print('abspath',os.path.abspath(os.curdir))
#print(inference.sliding_inference("samples/p001_001_Test1.wav","Stuttering is when my words get stuck, like I just did. Stuttering is not something that you can help, something that comes along with some people. I get worried like what other people might think if I do stutter. Kids should ignore the people who teases them about stuttering because it's not their fault. I woort about it when I'm on S words and they always get me stuck on stuff, on words with an S in the beginning. help us help those who stutter. Donate now to the Stuttering Foundation of America."))
print(inference.Inference("Please call Stella.","samples/p001_001_prolong.wav"))