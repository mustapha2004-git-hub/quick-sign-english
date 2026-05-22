import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import sounddevice as sd

print("Loading translation model...")
trans_tokenizer = AutoTokenizer.from_pretrained("BounharAbdelaziz/Terjman-Large-v2.0")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("BounharAbdelaziz/Terjman-Large-v2.0")

print("Loading TTS model...")
processor = SpeechT5Processor.from_pretrained("HAMMALE/speecht5-darija")
model = SpeechT5ForTextToSpeech.from_pretrained("HAMMALE/speecht5-darija")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

print("Testing Translation...")
input_ids = trans_tokenizer("Hello, how are you?", return_tensors="pt").input_ids
outputs = trans_model.generate(input_ids)
translated = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translated:", translated)

print("Testing TTS...")
inputs = processor(text=translated, return_tensors="pt")
speaker_embeddings = torch.zeros((1, 512))
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

print("Playing audio...")
sd.play(speech.numpy(), samplerate=16000)
sd.wait()
print("Success!")
