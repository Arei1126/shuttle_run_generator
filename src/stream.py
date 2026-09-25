import numpy as np
import soundfile as sf

SR = 44100
OUTPUT_FILE = "stream_test.wav"

# =========================================================
# 手続き型のストリーミング書き出し（最小限サンプル）
# =========================================================

# 1. 追記モード（"w" = write）でファイルを開く
# with ブロックを抜けるまでファイルは開かれっぱなしになり、メモリを節約できる
with sf.SoundFile(OUTPUT_FILE, mode="w", samplerate=SR, channels=1, subtype="PCM_16") as outfile:
	
	print(f"{OUTPUT_FILE} を開いたよ。追記を開始するね。")

	# 2. ダミーのチャンク（破片）を作るループ
	#   例として「1秒間のビープ音」を 5回（5チャンク）連続して書き出してみる
	for i in range(5):
		
		# (A) 1チャンク（今回は1秒）分の配列を都度作成する
		#     ここがシャトルランの「1往復分のバッファ」に相当するよ
		duration_sec = 1.0
		t = np.linspace(0, duration_sec, int(SR * duration_sec))
		
		# ちょっとずつ音が高くなるビープ音を生成
		freq = 440 + (i * 100) 
		chunk_data = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
		
		# (B) 作った配列をディスクへそのまま追記！
		#     書き出したら chunk_data のメモリは次のループで上書きされて消える
		outfile.write(chunk_data)
		
		print(f"チャンク {i+1}/5 を書き出したよ (周波数: {freq}Hz)")

print("すべてのストリーム書き込みが完了し、ファイルが安全に閉じられたよ！")
