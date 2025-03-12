import os
from functools import lru_cache
from typing import Union

import torch
import torchaudio
from huggingface_hub import hf_hub_download

import k2  # noqa
import sherpa
import sherpa_onnx
import numpy as np
from typing import Tuple
import wave

sample_rate = 16000


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and each sample should
        be 16-bit. Its sample rate does not need to be 16kHz.
    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples, which are
       normalized to the range [-1, 1].
       - sample rate of the wave file
    """

    with wave.open(wave_filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()  # it is in bytes
        num_samples = f.getnframes()
        samples = f.readframes(num_samples)
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32)

        samples_float32 = samples_float32 / 32768
        return samples_float32, f.getframerate()


def decode_offline_recognizer(
    recognizer: sherpa.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()

    s.accept_wave_file(filename)
    recognizer.decode_stream(s)

    text = s.result.text.strip()
    #  return text.lower()
    return text


def decode_online_recognizer(
    recognizer: sherpa.OnlineRecognizer,
    filename: str,
) -> str:
    samples, actual_sample_rate = torchaudio.load(filename)
    assert sample_rate == actual_sample_rate, (
        sample_rate,
        actual_sample_rate,
    )
    samples = samples[0].contiguous()

    s = recognizer.create_stream()

    tail_padding = torch.zeros(int(sample_rate * 0.3), dtype=torch.float32)
    s.accept_waveform(sample_rate, samples)
    s.accept_waveform(sample_rate, tail_padding)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    text = recognizer.get_result(s).text
    #  return text.strip().lower()
    return text.strip()


def decode_offline_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OfflineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)

    return s.result.text


def decode_online_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OnlineRecognizer,
    filename: str,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    s.accept_waveform(sample_rate, samples)

    tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    #  return recognizer.get_result(s).lower()
    return recognizer.get_result(s)


def decode(
    recognizer: Union[
        sherpa.OfflineRecognizer,
        sherpa.OnlineRecognizer,
        sherpa_onnx.OfflineRecognizer,
        sherpa_onnx.OnlineRecognizer,
    ],
    filename: str,
) -> str:
    if isinstance(recognizer, sherpa.OfflineRecognizer):
        return decode_offline_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa.OnlineRecognizer):
        return decode_online_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OfflineRecognizer):
        return decode_offline_recognizer_sherpa_onnx(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OnlineRecognizer):
        return decode_online_recognizer_sherpa_onnx(recognizer, filename)
    else:
        raise ValueError(f"Unknown recognizer type {type(recognizer)}")


@lru_cache(maxsize=30)
def get_pretrained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> Union[sherpa_onnx.OfflineRecognizer]:
    if repo_id in chinese_cantonese_english_japanese_korean_models:
        return chinese_cantonese_english_japanese_korean_models[repo_id](
            repo_id, decoding_method=decoding_method, num_active_paths=num_active_paths
        )
    else:
        raise ValueError(f"Unsupported repo_id: {repo_id}")


def _get_nn_model_filename(
    repo_id: str,
    filename: str,
    subfolder: str = "exp",
) -> str:
    nn_model_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return nn_model_filename


def _get_token_filename(
    repo_id: str,
    filename: str = "tokens.txt",
    subfolder: str = "data/lang_char",
) -> str:
    token_filename = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )
    return token_filename


@lru_cache(maxsize=10)
def _get_sense_voice_pre_trained_model(
    repo_id: str,
    decoding_method: str,
    num_active_paths: int,
) -> sherpa_onnx.OfflineRecognizer:
    assert repo_id in [
        "LPDoctor/ASR_model",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=nn_model,
        tokens=tokens,
        num_threads=2,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=True,
        use_itn=True,
    )

    return recognizer


chinese_cantonese_english_japanese_korean_models = {
    "LPDoctor/ASR_model": _get_sense_voice_pre_trained_model,
}

all_models = {
    **chinese_cantonese_english_japanese_korean_models,
}

language_to_models = {
    "SENSEVOICE": list(
        chinese_cantonese_english_japanese_korean_models.keys()
    ),
}