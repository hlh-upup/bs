from pydub import AudioSegment

# def mp3_to_wav(mp3_path, wav_path):
#     audio = AudioSegment.from_mp3(mp3_path)
#     audio.export(wav_path, format="wav")
#     print(f"转换完成：{wav_path}")

# # 用法示例
# mp3_to_wav("F:\\25digital\\实验室安全\\1去噪.MP3", "F:\\25digital\\实验室安全\\1去噪.wav")



from pydub import AudioSegment

def cut_audio_to_5s(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    cut_audio = audio[:3000]  # 前5秒，单位为毫秒
    cut_audio.export(output_path, format="wav")
    print(f"已剪切为5秒音频：{output_path}")

# 用法示例
cut_audio_to_5s("F:\\25digital\\实验室安全\\1去噪.wav", "F:\\25digital\\实验室安全\\1去噪_5s.wav")