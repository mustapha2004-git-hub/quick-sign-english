import os
from dotenv import load_dotenv
import torch
import sounddevice as sd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

print("Loading English to Darija Translation model (BounharAbdelaziz/Terjman-Large-v2.0)...")
trans_tokenizer = AutoTokenizer.from_pretrained("BounharAbdelaziz/Terjman-Large-v2.0", token=HF_TOKEN)
trans_model = AutoModelForSeq2SeqLM.from_pretrained("BounharAbdelaziz/Terjman-Large-v2.0", token=HF_TOKEN)

print("Loading Darija TTS model (HAMMALE/speecht5-darija)...")
tts_processor = SpeechT5Processor.from_pretrained("HAMMALE/speecht5-darija", token=HF_TOKEN)
tts_model = SpeechT5ForTextToSpeech.from_pretrained("HAMMALE/speecht5-darija", token=HF_TOKEN)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", token=HF_TOKEN)

speaker_embeddings = torch.zeros((1, 512))

def test_integration():
    english_text = "Hello, how are you?"
    print(f"Translating: '{english_text}'")
    
    # 1. Translate
    input_ids = trans_tokenizer(english_text, return_tensors="pt").input_ids
    outputs = trans_model.generate(input_ids)
    darija_text = trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Darija Translation: '{darija_text.encode('utf-8')}'")
    
    # 2. TTS
    print("Generating speech tensor...")
    tts_inputs = tts_processor(text=darija_text, return_tensors="pt")
    speech = tts_model.generate_speech(tts_inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    print(f"Speech generated successfully! Shape: {speech.shape}")

    # Not playing audio here since we just want to verify shapes and OOM
    print("Integration test passed!")

if __name__ == '__main__':
    test_integration()
