from pydub import AudioSegment

file_m4a = "example.m4a"
file_wav = "example.wav"

audio = AudioSegment.from_file(file_m4a, format="m4a")  # 打开 M4A 文件
audio.export(file_wav, format="wav")  # 保存为 wav 文件
