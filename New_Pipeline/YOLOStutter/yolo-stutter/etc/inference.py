import os
import sys
print(os.path.abspath(os.curdir))
os.chdir("D:\\stuff\\StutterProject\\New_Pipeline\\YOLOStutter\\yolo-stutter\\etc")
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)
import json
import torch
from torch import nn
import torchaudio
from torch.nn import functional as F

import utils.vits.commons as commons
import utils.vits.utils as utils
from utils.vits.models import SynthesizerTrn
from utils.vits.text.symbols import symbols
from utils.vits.text import text_to_sequence

from scipy.io.wavfile import write
from utils.vits.utils import load_wav_to_torch, load_filepaths_and_text
from utils.vits.mel_processing import spectrogram_torch


import importlib
importlib.reload(utils)
from torch.nn.utils.rnn import pad_sequence
from utils.model_utils.conv1d_transformer import Conv1DTransformerDecoder

# from utils.model_utils.dataset_5 import Dataset
from tqdm import tqdm
import wave

import logging

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

import numba.core.byteflow
numba_logger = logging.getLogger('numba.core.byteflow')
numba_logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cpu")
decoder = torch.load("../saved_models/decoder_tts_joint", map_location=device)
def print_full_tensor(tensor):
    """
    Print the entire content of a PyTorch tensor, regardless of shape.
    """
    torch.set_printoptions(threshold=float('inf'))
    print(tensor)
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_sample_rate_wave(audio_file_path):
    with wave.open(audio_file_path, 'rb') as wf:
        return wf.getframerate()


hps = utils.get_hparams_from_file("../utils/vits/configs/ljs_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("../saved_models/pretrained_ljs.pth", net_g, None)


def get_audio_a(filename):
    _sampling_rate = 22050 ####
    _max_wav_value = 32768.0
    _filter_length = 1024
    _hop_length = 256
    _win_length = 1024

    audio, sampling_rate = load_wav_to_torch(filename)
    if audio.numel() <= 1024:
        raise ValueError(f"Audio too short for spectrogram: {audio.numel()} samples")

    # print(sampling_rate)
    if sampling_rate != _sampling_rate:
        print("{} SR doesn't match target {} SR".format(
            sampling_rate, _sampling_rate))
        print("resampling to {} SR...".format(_sampling_rate))
        resampler=torchaudio.transforms.Resample(orig_freq=sampling_rate,new_freq=_sampling_rate)
        audio=resampler(audio)
    #flatten to 1 channel
    print(audio.shape)
    if audio.shape[0]>1:
        audio=audio.squeeze()
    '''normalized_waveform = audio / torch.max(torch.abs(audio))

    audio = normalized_waveform * 32767   # 2^15
    audio_norm = audio / _max_wav_value''' #DEPRECATED: corrupts the wav file
    audio = audio / audio.abs().max().clamp(min=1e-6)
    audio_norm = audio.unsqueeze(0)
    print('audio_norm:',audio_norm.shape)
    spec_filename = filename.replace(".wav", ".spec.pt")
    spec = spectrogram_torch(audio_norm, _filter_length,
        _sampling_rate, _hop_length, _win_length,
        center=False)

    spec = torch.squeeze(spec, 0)
    # torch.save(spec, spec_filename)  # save spec
    print('return fron audio_a')
    return spec, audio_norm


def get_labels(path):
    with open(path, "r") as f:
        labels = json.load(f)

    phonemes = labels[0]["phonemes"]
    text_path = path.replace("disfluent_labels", "gt_text")
    last = text_path.rfind('_')
    text_path = text_path[:last] + ".txt"
    with open(text_path, 'r') as file:
        text = file.read()

    # text = labels[0]["text"]

    for w in phonemes:
        w["start"] = int(w["start"] / 0.016)
        w["end"] = int(w["end"] / 0.016)

    return phonemes, text

def get_audio(path):
    spec, wav = get_audio_a(path) #[spec is 1, d, t]

    return spec, wav

def process_audio(spec, wav, _text):

    stn_tst = get_text(_text, hps)
    # stn_tst = get_text("knows both he them", hps)
    # stn_tst = get_text("I miss you", hps)
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

    y = spec.unsqueeze(0)
    y_lengths = torch.LongTensor([y.shape[-1]])
    t = net_g(x_tst, x_tst_lengths, y, y_lengths)
    # print(len(t[2]))
    o, l_length, (neg_cent, attn), ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x_tst, x_tst_lengths, y, y_lengths) #phoneme, mel_spec respectively

    # print("neg_cent shape: ", neg_cent.shape)

    neg_cent = neg_cent.squeeze(0)

    neg_cent = nn.functional.softmax(neg_cent, dim=1)

    return neg_cent


def get_soft_attention(hps, net_g, text, text_lengths, spec, spec_lengths):
    with torch.no_grad():
        print(text.shape, text_lengths.shape, spec.shape, spec_lengths.shape)
        o, l_length, (neg_cent, attn), ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text,
                                                                                                             text_lengths,
                                                                                                             spec,
                                                                                                             spec_lengths)  # phoneme, mel_spec respectively

        neg_cent = nn.functional.softmax(neg_cent, dim=-1)

    return neg_cent


def single_inference(hps, wav_path, ref_text, downsample_factor, decoder, device):
    text = get_text(ref_text, hps)
    text_length = torch.tensor(text.size(0))

    spec, wav = get_audio_a(wav_path)  # [1, d, t]
    spec_length = torch.tensor(spec.shape[-1])

    text = text.unsqueeze(0)
    spec = spec.unsqueeze(0)
    text_length = text_length.unsqueeze(0)
    spec_length = spec_length.unsqueeze(0)

    # print("text: ", text.shape) # [1, U]
    # print("spec: ", spec.shape) # [1, 513, T]
    # print(text_length)
    # print(spec_length)

    num_regions = int(spec_length // downsample_factor)

    text = text.to(device)
    text_length = text_length.to(device)
    spec = spec.to(device)
    spec_length = spec_length.to(device)

    soft_attention = get_soft_attention(
        hps, net_g, text, text_length, spec, spec_length
    )

    orig_text_dim_shape = soft_attention.shape[-1]

    new_soft_attention = nn.functional.pad(
        soft_attention,
        (
            0,
            768 - soft_attention.shape[-1],
            0,
            1024 - soft_attention.shape[-2],
        ),
    )

    mask = torch.ones((1, 64), dtype=torch.bool)
    mask[0, : num_regions + 1] = False

    output = decoder(new_soft_attention, mask.to(device))

    return output


def Inference(ref_text,wav_path):
    global net_g

    text_channels = 768
    kernel_size= 3
    kernel_stride = 1
    num_blocks = 4
    num_classes = 5   ## change
    downsample_factor = 16  #8
    n_heads = 8
    n_layers = 8




    labels = ["rep", "block", "missing", "replace", "prolong"]

    net_g = net_g.to(device)

    #ref_text = "Please call Stella."
    #wav_path = "samples/p001_001_rep.wav"

    output = single_inference(hps, wav_path, ref_text, downsample_factor, decoder, device)
    print_full_tensor(output)
    '''disfluency_type_pred = output[:, :, 3:]
    
    type_pred_softmax = torch.log_softmax(disfluency_type_pred, dim=-1)
    a, y_pred_labels = torch.max(type_pred_softmax, dim=-1)

    # disfluency_bound_pred = disf_out[:, :, :2]
    disfluency_bound_pred = output[:, :, :2]
    disfluency_bound_pred = disfluency_bound_pred.squeeze(0)

    disf_words_pred = []
    bounds_pred = disfluency_bound_pred[0] * 1024 * 256 / 22050
    print(bounds_pred)
    print({"start": bounds_pred[0].item(), "end": bounds_pred[1].item(), "type": labels[y_pred_labels[0][0].item()]})
    return {"start": bounds_pred[0].item(), "end": bounds_pred[1].item(), "type": labels[y_pred_labels[0][0].item()]}
'''
    disfluency_bound_pred = output[:, :, :2].squeeze(0)  # [num_regions, 2]
    disfluency_type_pred = output[:, :, 3:]  # [1, num_regions, num_classes]
    type_pred_softmax = torch.softmax(disfluency_type_pred, dim=-1).squeeze(0)  # [num_regions, num_classes]
    print('type_pred_softmax')
    print_full_tensor(type_pred_softmax)
    conf, y_pred_labels = torch.max(type_pred_softmax, dim=-1)  # [num_regions]
    print(conf)
    # Convert time units
    time_scale = 1024 * 256 / 22050

    # Confidence threshold (tune this)
    conf_thresh = 0.9

    results = []
    for i in range(disfluency_bound_pred.shape[0]):
        if conf[i] > conf_thresh:
            start = (disfluency_bound_pred[i][0] * time_scale).item()
            end = (disfluency_bound_pred[i][1] * time_scale).item()
            dis_type = labels[y_pred_labels[i].item()]
            results.append({
                "start": start,
                "end": end,
                "type": dis_type,
                "confidence": conf[i].item()
            })

    # Sort by start time
    results = sorted(results, key=lambda x: x["start"])

    print(f"Detected {len(results)} stuttering regions:")
    for r in results:
        print(f"  {r}")

    return results


def sliding_inference(wav_path, ref_text, chunk_sec=10, overlap=0.5):
    sr_target = 22050
    chunk_samples = int(chunk_sec * sr_target)
    hop = int(chunk_samples * (1 - overlap))
    min_samples = 1024  # must match STFT window minimum

    # Load audio
    audio, sr = load_wav_to_torch(wav_path)  # [C, T]
    print("RAW audio shape:", audio.shape, "SR:", sr)
    if audio.dim() == 2 and audio.shape[1] == 2:
        audio = audio.transpose(0, 1)
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0)  # [T]

    # Resample if needed
    if sr != sr_target:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr_target)
        audio = resampler(audio.unsqueeze(0)).squeeze(0)
        print("Resampled audio shape:", audio.shape)

    # Normalize and clamp
    audio = audio / audio.abs().max().clamp(min=1e-6)
    audio = audio.clamp(-1.0, 1.0)
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    results = []

    for i in range(0, len(audio), hop):
        chunk = audio[i:i + chunk_samples]

        # Skip too short chunks
        if chunk.numel() < min_samples:
            continue

        # Prepare chunk for VITS
        chunk = chunk.unsqueeze(0)  # [1, T]
        chunk = chunk.float()

        # Save temporary wav for debugging / inference
        torchaudio.save(
            "tmp_chunk.wav",
            chunk,
            sr_target,
            encoding="PCM_S",
            bits_per_sample=16
        )

        # Safe inference
        try:
            chunk_results = Inference(ref_text, "tmp_chunk.wav")
        except Exception as e:
            print(f"Skipping chunk {i}:{i + chunk_samples} due to error: {e}")
            continue

        # Offset results to original timeline
        offset = i / sr_target
        for r in chunk_results:
            r["start"] += offset
            r["end"] += offset
            results.append(r)

    # Sort results by start time
    results = sorted(results, key=lambda x: x["start"])
    print(f"Detected {len(results)} stuttering regions:")
    for r in results:
        print(f"  {r}")

    return results
