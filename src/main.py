# オクターブ表記は一つ上を書いてしまっている。ミス
import os
import numpy as np
import soundfile as sf
import math
from scipy.signal import resample_poly
import csv
from typing import Dict, Any, List, Optional

SR = 36000
#SR = 44100
Note = {}
path_tone = "../material/tone/"
path_output = "output.flac"
path_count = "../material/voice/"
path_five = "../material/5sec.wav"
path_start = "../material/start.wav"

#path_levels = "./levels.csv"
path_levels = "./level_test.csv"

Now = {
    "count": 0,
    "level": 0,
    "time": 0,
}


Levels = None

def db_to_gain(db: float) -> float:
	"""デシベル（dB）をリニア倍率に変換する"""
	return 10.0 ** (db / 20.0)
MASTER_GAIN = db_to_gain(1)

def load_csv_as_indexed_list(filepath: str, id_column: str) -> List[Optional[Dict[str, str]]]:
	"""
	連続する整数IDに基づいて、インデックスがIDと一致するリストを生成する。
	IDが1始まりの場合、インデックス0には None を配置してズレを防ぐ。
	"""
	with open(filepath, mode="r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		# 0番インデックス用のダミーを初期配置
		result: List[Optional[Dict[str, str]]] = [None]
		
		for expected_id, row in enumerate(reader, start=1):
			current_id = int(row[id_column])
			
			# 連続性のバリデーション（データ破損や欠番の早期検知）
			if current_id != expected_id:
				raise ValueError(
					f"IDの連続性が崩れています。期待値: {expected_id}, 実際: {current_id}"
				)
			result.append(dict(row))
			
	return result

def load_csv_with_key(filepath: str, key_column: str) -> Dict[str, Dict[str, str]]:
        result: Dict[str, Dict[str, str]] = {}
        with open(filepath, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_key = row[key_column]
                if record_key in result:
                    raise ValueError(f"重複キーを検出しました: {record_key}")
                result[record_key] = dict(row)
        return result



def _append_count(data, count,duration):
        offset = duration/18 *SR
        audio = load_wav(f"{path_count}{count}.wav")
        data[int(offset):int(offset + len(audio))] += audio[:len(audio)]

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

def append_count(f, duration_sec, carry, count):
    audio = load_wav(f"{path_count}{count}.wav")
    
    offset_plus = int(duration_sec/9 * SR * 1)
    tail_sec = 3.0
    tail_samples = int(tail_sec * SR)
    buffer_len = len(audio) + tail_samples + offset_plus
    chunk_data = np.zeros(buffer_len, dtype=np.float32)
    
    if carry is not None and len(carry) > 0:
        overlap_len = min(len(carry), buffer_len)
        chunk_data[:overlap_len] += carry[:overlap_len]
    
    offset = int(duration_sec/18 * 0) +  offset_plus
    chunk_data[offset:offset+len(audio)] += audio[:len(audio)]

    f.write(chunk_data)

    next_carry = chunk_data[offset+len(audio):]
    return next_carry

def append_up(f, duration_sec: float, carry: np.ndarray, count= None, final=False) -> np.ndarray:
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
    
    if count:
        _append_count(chunk_data, count, duration_sec)

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


    if not final:
        # 3. 8拍目の重ね合わせ
        bass_start = int((8 * spb) * SR)
        notes = ["c3", "c4"]
        for bass in notes:
            audio = Note[bass]
            end = bass_start + len(audio)
            use_len = min(end, buffer_len) - bass_start
            if use_len > 0:
                chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:lap_samples] * MASTER_GAIN)

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry

def append_down(f, duration_sec: float, carry: np.ndarray, count = None, final=False) -> np.ndarray:
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

    if count:
        _append_count(chunk_data, count, duration_sec)

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
    if not final:
        bass_start = int((8 * spb) * SR)
        notes = ["c3", "c4"]
        for bass in notes:
            audio = Note[bass]
            end = bass_start + len(audio)
            use_len = min(end, buffer_len) - bass_start
            if use_len > 0:
                chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:lap_samples] * MASTER_GAIN)

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry


def append_up_lvup(f, duration_sec: float, next_duration_sec:float, carry: np.ndarray, count = None) -> np.ndarray:
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

    if count:
        _append_count(chunk_data, count, duration_sec)

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
    f.write(chunk_data[:lap_samples] * MASTER_GAIN)

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry

def append_down_lvup(f, duration_sec: float, next_duration_sec:float, carry: np.ndarray, count = None) -> np.ndarray:
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

    if count:
        _append_count(chunk_data, count, duration_sec)

    # 2. 音階の配置（枠で切り詰めず、音源本来の長さをそのまま重ねる）
    up = ["c5", "b4", "a4", "g4", "f4", "e4", "d4", "c4"]
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
    f.write(chunk_data[:lap_samples] * MASTER_GAIN)

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry


def append_lvup(f, duration_sec: float, carry: np.ndarray) -> np.ndarray:
    spb = duration_sec/9
    samples = int(spb*SR)
    tail = int(3*SR)
    buffer_len = samples + tail
    chunk_data = np.zeros(buffer_len, dtype=np.float32)

    # 1. 前回の往復から持ち越された余韻を先頭に合流
    if carry is not None and len(carry) > 0:
        overlap_len = min(len(carry), buffer_len)
        chunk_data[:overlap_len] += carry[:overlap_len]

    bass_start = int(0)
    for bass in ["c3", "c4", "c5"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    bass_start = int((spb/4 *SR))
    for bass in ["g4", "g5"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]

    bass_start = int(2*(spb/4 *SR))

    for bass in ["d5", "d6"]:
        audio = Note[bass]
        end = bass_start + len(audio)
        use_len = min(end, buffer_len) - bass_start
        if use_len > 0:
            chunk_data[bass_start:bass_start + use_len] += audio[:use_len]
    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:samples])

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[samples:]
    return next_carry


def five_count(f) -> np.ndarray:
    samples = int(5*SR)
    tail = int(3*SR)
    buffer_len = samples + tail
    chunk_data = np.zeros(buffer_len, dtype=np.float32)

    audio = load_wav(path_five)
    #chunk_data[0:len(audio)] += audio[:len(audio)]

    copy_len = min(len(audio), buffer_len)
    chunk_data[:copy_len] += audio[:copy_len]

    for i in range(3):
        start = (i+2)*SR
        audio = load_wav(f"{path_count}{3-i}.wav")
        chunk_data[start:start + len(audio)] += audio[:len(audio)]


    # 4. 今回の確定分（lap_samples）だけをファイルに追記
    f.write(chunk_data[:samples])

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[samples:]
    return next_carry


def append_start(f, duration_sec: float, carry: np.ndarray) -> np.ndarray:
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

    # スタート音声
    audio = load_wav(path_start)
    offset = int(spb/2 * SR)
    chunk_data[offset:offset+len(audio)] += audio[:len(audio)]


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
    f.write(chunk_data[:lap_samples] * MASTER_GAIN)

    # 5. はみ出した余韻を「次の回」のために返す
    next_carry = chunk_data[lap_samples:]
    return next_carry

def main():
    # pre process
    notes = ["c3", "c4", "d4", "e4", "f4", "g4", "a4", "b4", "c5", "d5", "g5", "c6", "d6"]
    for note in notes:
        filepath = f"{path_tone}{note}.wav"
        if os.path.exists(filepath):
            Note[note] = load_wav(filepath)
    
    Levels = load_csv_as_indexed_list(path_levels, "level")
    max_level = len(Levels) - 1
    print(f"max:{max_level}")

    #with sf.SoundFile(path_output, mode="w", samplerate=SR, channels=1, subtype="PCM_16") as f:
    with sf.SoundFile(path_output, mode="w", samplerate=SR, channels=1, format="FLAC") as f:
        
        start_mute = int(1.5*SR)
        chunk_data = np.zeros(start_mute, dtype=np.float32)
        f.write(chunk_data)

        first_duration_sec = 0.02*60*60 / float(Levels[1]["speed"])
        
        carry = five_count(f)
        carry = append_lvup(f, duration_sec=first_duration_sec, carry=carry)
        carry = append_start(f,duration_sec=first_duration_sec, carry=carry)

        count = 1
        for i in range(1, len(Levels)):
            print(f"level:{i}")
            target_count = int(Levels[i]["count_in_level"]) + count - 1

            spd = float(Levels[i]["speed"])
            duration_sec = 0.02*60*60 / spd


            if i != max_level:      # 通常
                while(count < target_count):
                    is_odd = (count % 2 == 0)
                    if is_odd:
                        carry = append_up(f, duration_sec=duration_sec, carry=carry, count=count)
                    else:
                        carry = append_down(f, duration_sec=duration_sec, carry=carry, count=count)
                    count = count + 1

                next_duration_sec = 0.02*60*60/ float(Levels[i+1]["speed"])
                is_odd = (count % 2 == 0)
                if is_odd:
                    carry = append_up_lvup(f, duration_sec=duration_sec, next_duration_sec=next_duration_sec, carry=carry, count=count)
                else:
                    carry = append_down_lvup(f, duration_sec=duration_sec, next_duration_sec=next_duration_sec, carry=carry, count=count)
                count = count + 1
            else: # ファイルの最後
                while(count < target_count-1):
                    is_odd = (count % 2 == 0)
                    if is_odd:
                        carry = append_up(f, duration_sec=duration_sec, carry=carry, count=count)
                    else:
                        carry = append_down(f, duration_sec=duration_sec, carry=carry, count=count)
                    count = count + 1
                is_odd = (count % 2 == 0)
                if is_odd:
                    carry = append_up(f, duration_sec=duration_sec, carry=carry, count=count, final = True)
                else:
                    carry = append_down(f, duration_sec=duration_sec, carry=carry, count=count, final = True)
                # 最後のカウントだけ
                count = count + 1
                carry = append_count(f,duration_sec=duration_sec, carry=carry, count=count)

        if carry is not None:
            f.write(carry)

if __name__ == "__main__":
    main()
