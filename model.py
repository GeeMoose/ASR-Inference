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
    device=None,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    
    # 如果使用 GPU，将样本转换为 CUDA 张量进行预处理
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        # 将样本转换为 torch 张量并移至 GPU
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
        # 可以在这里添加 GPU 上的预处理操作
        # 处理完后转回 numpy
        samples = samples_tensor.cpu().numpy()
    
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(s)

    return s.result.text


def decode_online_recognizer_sherpa_onnx(
    recognizer: sherpa_onnx.OnlineRecognizer,
    filename: str,
    device=None,
) -> str:
    s = recognizer.create_stream()
    samples, sample_rate = read_wave(filename)
    
    # 如果使用 GPU，将样本转换为 CUDA 张量进行预处理
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
        # 可以在这里添加 GPU 上的预处理操作
        samples = samples_tensor.cpu().numpy()
    
    s.accept_waveform(sample_rate, samples)

    tail_paddings = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)

    return recognizer.get_result(s)


def decode(
    recognizer: Union[
        sherpa.OfflineRecognizer,
        sherpa.OnlineRecognizer,
        sherpa_onnx.OfflineRecognizer,
        sherpa_onnx.OnlineRecognizer,
    ],
    filename: str,
    device=None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(recognizer, sherpa.OfflineRecognizer):
        return decode_offline_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa.OnlineRecognizer):
        return decode_online_recognizer(recognizer, filename)
    elif isinstance(recognizer, sherpa_onnx.OfflineRecognizer):
        return decode_offline_recognizer_sherpa_onnx(recognizer, filename, device)
    elif isinstance(recognizer, sherpa_onnx.OnlineRecognizer):
        return decode_online_recognizer_sherpa_onnx(recognizer, filename, device)
    else:
        raise ValueError(f"Unknown recognizer type {type(recognizer)}")


@lru_cache(maxsize=30)
def get_pretrained_model(
    repo_id: str,
    decoding_method: str = "greedy_search",
    num_active_paths: int = 4,
    device=None,
) -> Union[sherpa_onnx.OfflineRecognizer]:
    """Get a pretrained model from huggingface.co"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if repo_id in chinese_cantonese_english_japanese_korean_models:
        return chinese_cantonese_english_japanese_korean_models[repo_id](
            repo_id, 
            decoding_method=decoding_method, 
            num_active_paths=num_active_paths,
            device=device  # 传递设备参数
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
    device=None,
) -> sherpa_onnx.OfflineRecognizer:
    # 检查是否使用 GPU
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        # 设置环境变量以使用 CUDA
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
        os.environ["ORT_CUDA_PROVIDER_OPTIONS"] = "device_id=0;cudnn_conv_algo_search=EXHAUSTIVE;cudnn_conv_use_max_workspace=1"
        # 设置 ONNX 运行时环境变量
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        print("使用 CUDA 进行推理，设备：", device)
    else:
        # 清除环境变量
        if "ORT_TENSORRT_FP16_ENABLE" in os.environ:
            del os.environ["ORT_TENSORRT_FP16_ENABLE"]
        if "ORT_CUDA_PROVIDER_OPTIONS" in os.environ:
            del os.environ["ORT_CUDA_PROVIDER_OPTIONS"]
        print("使用 CPU 进行推理")

    # 其余代码保持不变
    assert repo_id in [
        "LPDoctor/ASR_model",
    ], repo_id

    nn_model = _get_nn_model_filename(
        repo_id=repo_id,
        filename="model.onnx",
        subfolder=".",
    )

    tokens = _get_token_filename(repo_id=repo_id, subfolder=".")

    # 移除 provider_options 参数
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=nn_model,
        tokens=tokens,
        num_threads=2 if device is None or device.type == "cpu" else 1,  # GPU 模式下减少 CPU 线程
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