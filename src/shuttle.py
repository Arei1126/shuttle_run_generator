import csv
import os
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf

# ==============================================================================
# グローバル設定 & 共通定数
# ==============================================================================
SAMPLE_RATE = 44100
ASSET_DIR = "assets"		# 音声ファイルを置くフォルダ
CSV_PATH = "levels.csv"		# 仕様CSVのパス
OUTPUT_PATH = "shuttle_run.wav"	# 出力先パス

# タイムラインイベント型: (再生開始秒数, アセット識別キー)
TimelineEvent = Tuple[float, str]


# ==============================================================================
# 基盤モジュール（WAVロード・ミキシング・CSV読込）※基本的に変更不要
# ==============================================================================
class AudioAssetManager:
	"""WAVファイルの読み込み・キャッシュおよび波形合成を担うクラス"""

	def __init__(self, asset_dir: str, sample_rate: int = 44100):
		self.asset_dir = asset_dir
		self.sample_rate = sample_rate
		self.cache: Dict[str, np.ndarray] = {}

	def load_sample(self, key: str, filename: str) -> np.ndarray:
		"""単一のWAVをロードしてモノラルのfloat32配列として保持"""
		filepath = os.path.join(self.asset_dir, filename)
		if not os.path.exists(filepath):
			raise FileNotFoundError(f"アセットが見つかりません: {filepath}")

		data, sr = sf.read(filepath)
		if sr != self.sample_rate:
			raise ValueError(f"サンプリングレート不一致 ({filepath}): {sr} != {self.sample_rate}")

		# ステレオ音源の場合はモノラル化
		if data.ndim > 1:
			data = np.mean(data, axis=1)

		audio = data.astype(np.float32)
		self.cache[key] = audio
		return audio

	def register_memory_asset(self, key: str, audio: np.ndarray):
		"""メモリ上で生成した波形を直接登録"""
		self.cache[key] = audio.astype(np.float32)

	def get_duration(self, key: str) -> float:
		"""登録済みアセットの長さ（秒数）を取得"""
		if key not in self.cache:
			raise KeyError(f"未ロードのアセットです: {key}")
		return len(self.cache[key]) / self.sample_rate

	def render_timeline(self, timeline: List[TimelineEvent], margin_sec: float = 3.0) -> np.ndarray:
		"""
		タイムライン（秒数とキーのペア）を基に、巨大キャンバスへ加算合成する
		"""
		if not timeline:
			return np.zeros(0, dtype=np.float32)

		# 全体の所要時間を算出（最後のイベント開始秒 + その音の長さ + 余韻マージン）
		max_time = 0.0
		for start_sec, key in timeline:
			if key in self.cache:
				end_time = start_sec + self.get_duration(key)
				if end_time > max_time:
					max_time = end_time

		total_samples = int(np.ceil((max_time + margin_sec) * self.sample_rate))
		master_buffer = np.zeros(total_samples, dtype=np.float32)

		# 加算合成（ミキシング）
		for start_sec, key in timeline:
			if key not in self.cache:
				print(f"Warning: アセットキー '{key}' がロードされていないためスキップします。")
				continue

			audio = self.cache[key]
			start_idx = int(start_sec * self.sample_rate)
			end_idx = start_idx + len(audio)

			if start_idx < len(master_buffer):
				usable_len = min(end_idx, len(master_buffer)) - start_idx
				master_buffer[start_idx:start_idx + usable_len] += audio[:usable_len]

		# 音割れ防止（ピークノーマライズ）
		peak = np.max(np.abs(master_buffer))
		if peak > 1.0:
			master_buffer /= peak

		return master_buffer


def read_levels_csv(csv_path: str) -> List[dict]:
	"""CSVファイルを読み込み、扱いやすい辞書リストに変換"""
	records = []
	with open(csv_path, mode="r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			records.append({
				"level": int(row["level"]),
				"speed": float(row["speed"]),
				"laps": int(row["laps"])
			})
	return records


# ==============================================================================
# カスタマイズ領域（ここを自由に書き換えて演出や音の配置を調整してね）
# ==============================================================================

def schedule_lap_events(
	start_sec: float,
	lap_sec: float,
	lap_num: int,
	is_level_up: bool
) -> List[TimelineEvent]:
	"""
	1往復の中身をどう鳴らすかを決める関数
	:param start_sec: この往復が始まる秒数（現在の再生ヘッド）
	:param lap_sec: この1往復の所要時間（72 / speed）
	:param lap_num: 通算往復回数（カウント用）
	:param is_level_up: この往復でレベルが切り替わるか
	"""
	events: List[TimelineEvent] = []

	if not is_level_up:
		# --- 通常往復: 7音階の配置例 ---
		step = lap_sec / 7.0
		for i in range(7):
			events.append((start_sec + i * step, f"note_{i + 1}"))

		# 折り返しライン到達時のカウント音声
		events.append((start_sec + 7 * step, f"count_{lap_num}"))

	else:
		# --- レベルアップ時: チャイム4音の配置例 ---
		step = (lap_sec * 0.7) / 4.0
		for i in range(4):
			events.append((start_sec + i * step, f"chime_{i + 1}"))

		# カウント音声（往復終了の少し手前など、自由に調整可）
		events.append((start_sec + lap_sec - 0.5, f"count_{lap_num}"))

	return events


def build_full_timeline(
	levels: List[dict],
	countdown_duration: float
) -> List[TimelineEvent]:
	"""
	カウントダウンから全レベル終了までのタイムラインを構築する
	"""
	timeline: List[TimelineEvent] = []

	# 1. 冒頭のカウントダウン音声を配置
	timeline.append((0.0, "countdown"))

	# 2. カウントダウン終了後を基準時刻としてポインタを開始
	current_time = countdown_duration
	total_laps = 0

	# 3. CSVの各レベルを走査
	for lvl in levels:
		speed = lvl["speed"]
		laps = lvl["laps"]
		level_idx = lvl["level"]

		# 20m往復の所要時間 (秒): 20m / (speed * 1000 / 3600)
		lap_sec = 72.0 / speed

		for lap in range(laps):
			total_laps += 1
			# レベル2以上の初めの1往復目をレベルアップ扱いとする例
			is_level_up = (lap == 0 and level_idx > 1)

			# 1往復分のイベントを取得して追加
			lap_events = schedule_lap_events(
				start_sec=current_time,
				lap_sec=lap_sec,
				lap_num=total_laps,
				is_level_up=is_level_up
			)
			timeline.extend(lap_events)

			# ポインタを次の往復開始時刻へ進める
			current_time += lap_sec

	return timeline


# ==============================================================================
# エントリポイント
# ==============================================================================
if __name__ == "__main__":
	# 1. マネージャーの初期化
	mgr = AudioAssetManager(asset_dir=ASSET_DIR, sample_rate=SAMPLE_RATE)

	# --- アセットのロード例 ---
	# ※手元のファイル名に合わせて書き換えてね
	# mgr.load_sample("countdown", "countdown.wav")
	# for i in range(1, 8):
	# 	mgr.load_sample(f"note_{i}", f"note_{i}.wav")
	# for i in range(1, 5):
	# 	mgr.load_sample(f"chime_{i}", f"chime_{i}.wav")
	# for i in range(1, 101):
	# 	mgr.load_sample(f"count_{i}", f"count_{i}.wav")

	# （テスト用ダミー：ファイルがない場合でも動くようにサイン波を登録）
	t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3))
	mgr.register_memory_asset("countdown", np.sin(2 * np.pi * 440 * np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0))))
	for i, f in enumerate([261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9]):
		mgr.register_memory_asset(f"note_{i+1}", np.sin(2 * np.pi * f * t) * np.exp(-3 * t))
	for i, f in enumerate([523.3, 659.3, 784.0, 1046.5]):
		mgr.register_memory_asset(f"chime_{i+1}", np.sin(2 * np.pi * f * t) * np.exp(-2 * t))
	for i in range(1, 50):
		mgr.register_memory_asset(f"count_{i}", np.sin(2 * np.pi * 880 * t) * 0.5)

	# 2. CSV読み込み
	# levels = read_levels_csv(CSV_PATH)
	levels = [
		{"level": 1, "speed": 8.5, "laps": 7},
		{"level": 2, "speed": 9.0, "laps": 8},
	]

	# 3. タイムライン構築
	countdown_len = mgr.get_duration("countdown")
	timeline = build_full_timeline(levels, countdown_duration=countdown_len)

	# 4. レンダリング & 書き出し
	print("音声レンダリング中...")
	output_audio = mgr.render_timeline(timeline)
	sf.write(OUTPUT_PATH, output_audio, SAMPLE_RATE)
	print(f"完了！ファイルを出力したよ: {OUTPUT_PATH}")
