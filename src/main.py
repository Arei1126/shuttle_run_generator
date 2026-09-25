# オクターブ表記は一つ上を書いてしまっている。ミス
import os
import numpy as np
import soundfile as sf
import math
from scipy.signal import resample_poly

SR = 44100
Note = {}
path_tone = "../material/tone/"
path_output = "output.wav"

Now = {
    "count": 0,
    "level": 0,
    "time": 0,
}

def load_wav(path: str) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # サンプリングレートが異なる場合のみリサンプリングを実行
    if sr != SR:
        gcd = math.gcd(SR, sr)
        up = SR // gcd
        down = sr // gcd
        data = resample_poly(data, up=up, down=down).astype(np.float32)

    return data.astype(np.float32)

def append_up(f, duration_sec: float, carry: np.ndarray) -> np.ndarray:
    """
    1往復分の音声を書き出し、余った余韻（carry）を次へ引き継ぐ
    :param f: SoundFileのファイルオブジェクト
    :param duration_sec: 1往復の所要秒数
    :param carry: 前回の往復からはみ出した余韻配列
    :return: 次回の往復へ持ち越す余韻配列
    """
    lap_samples = int(duration_sec * SR)
    spb = duration_sec / 9.0

    # 余韻を受け止めるため、作業バッファを長めに確保（例: +3秒分）
    tail_sec = 3.0
    tail_samples = int(tail_sec * SR)
    buffer_len = lap_samples + tail_samples
    chunk_data = np.zeros(buffer_len, dtype=np.float32)

    # 1. 前回の往復から持ち越された余韻を先頭に合流
    if carry is not None and len(carry) > 0:
        overlap_len = min(len(carry), buffer_len)
        chunk_data[:overlap_len] += carry[:overlap_len]

    # 2. 音階の配置（枠で切り詰めず、音源本来の長さをそのまま重ねる）
    up = ["c4", "d4", "e4", "f4", "g4", "a4", "b4", "c5"]
    for i, note in enumerate(up):
        audio = Note[note]
        start = int((i * spb) * SR)
        end = start + len(audio)

        use_len = min(end, buffer_len) - start
        if use_len > 0:
            chunk_data[start:start + use_len] += audio[:use_len]

    # 3. 8拍目の重ね合わせ
    bass_start = int((8 * spb) * SR)
    for bass in ["c3", "c4"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:lap_samples])

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry

def append_down(f, duration_sec: float, carry: np.ndarray) -> np.ndarray:
    """
    1往復分の音声を書き出し、余った余韻（carry）を次へ引き継ぐ
    :param f: SoundFileのファイルオブジェクト
    :param duration_sec: 1往復の所要秒数
    :param carry: 前回の往復からはみ出した余韻配列
    :return: 次回の往復へ持ち越す余韻配列
    """
    lap_samples = int(duration_sec * SR)
    spb = duration_sec / 9.0

    # 余韻を受け止めるため、作業バッファを長めに確保（例: +3秒分）
    tail_sec = 3.0
    tail_samples = int(tail_sec * SR)
    buffer_len = lap_samples + tail_samples
    chunk_data = np.zeros(buffer_len, dtype=np.float32)

    # 1. 前回の往復から持ち越された余韻を先頭に合流
    if carry is not None and len(carry) > 0:
        overlap_len = min(len(carry), buffer_len)
        chunk_data[:overlap_len] += carry[:overlap_len]

    # 2. 音階の配置（枠で切り詰めず、音源本来の長さをそのまま重ねる）
    up = ["c5", "b4", "a4", "g4", "f4", "e4", "d4", "c4"]
    for i, note in enumerate(up):
        audio = Note[note]
        start = int((i * spb) * SR)
        end = start + len(audio)

        use_len = min(end, buffer_len) - start
        if use_len > 0:
            chunk_data[start:start + use_len] += audio[:use_len]

    # 3. 8拍目の重ね合わせ
    bass_start = int((8 * spb) * SR)
    for bass in ["c3", "c4"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:lap_samples])

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry


def append_up_lvup(f, duration_sec: float, next_duration_sec:float, carry: np.ndarray) -> np.ndarray:
    """
    1往復分の音声を書き出し、余った余韻（carry）を次へ引き継ぐ
    :param f: SoundFileのファイルオブジェクト
    :param duration_sec: 1往復の所要秒数
    :param carry: 前回の往復からはみ出した余韻配列
    :return: 次回の往復へ持ち越す余韻配列
    """
    lap_samples = int(duration_sec * SR)
    spb = duration_sec / 9.0
    nspb = next_duration_sec/9.0
    lap_samples = int(spb*8*SR + nspb*SR) 

    # 余韻を受け止めるため、作業バッファを長めに確保（例: +3秒分）
    tail_sec = 3.0
    tail_samples = int(tail_sec * SR)
    buffer_len = lap_samples + tail_samples
    chunk_data = np.zeros(buffer_len, dtype=np.float32)

    # 1. 前回の往復から持ち越された余韻を先頭に合流
    if carry is not None and len(carry) > 0:
        overlap_len = min(len(carry), buffer_len)
        chunk_data[:overlap_len] += carry[:overlap_len]

    # 2. 音階の配置（枠で切り詰めず、音源本来の長さをそのまま重ねる）
    up = ["c4", "d4", "e4", "f4", "g4", "a4", "b4", "c5"]
    for i, note in enumerate(up):
        audio = Note[note]
        start = int((i * spb) * SR)
        end = start + len(audio)

        use_len = min(end, buffer_len) - start
        if use_len > 0:
            chunk_data[start:start + use_len] += audio[:use_len]

    # 3. 8拍目チャイム（加速）
    bass_start = int((8 * spb) * SR)
    for bass in ["c3", "c4", "c5"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    bass_start = int((8 * spb) * SR + (nspb/4 *SR))
    for bass in ["g4", "g5"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]
    bass_start = int((8 * spb) * SR + 2*(nspb/4 *SR))

    for bass in ["d5", "d6"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:lap_samples])

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry




def main():
    notes = ["c3", "c4", "d4", "e4", "f4", "g4", "a4", "b4", "c5", "d5", "g5", "c6", "d6"]
    for note in notes:
        filepath = f"{path_tone}{note}.wav"
        if os.path.exists(filepath):
            Note[note] = load_wav(filepath)

    with sf.SoundFile(path_output, mode="w", samplerate=SR, channels=1, subtype="PCM_16") as f:
        carry = None

        # ループで回すときは、carry をバトンタッチしていくだけ！
        # （例として 9秒の往復を3回連続で鳴らす場合）
        carry = append_up(f, duration_sec=9.0, carry=carry)
        carry = append_down(f, duration_sec=9.0, carry=carry)
        carry = append_up_lvup(f, duration_sec=9.0, next_duration_sec=8, carry=carry)
        #carry = append_up(f, duration_sec=6.0, carry=carry)
        #carry = append_up(f, duration_sec=3.0, carry=carry)

        # すべての往復が終わったあと、最後に残った余韻を書き出して綺麗に終わる
        if carry is not None:
            f.write(carry)

if __name__ == "__main__":
    main()
