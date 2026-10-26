import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "Create a high-energy electronic dance music (EDM) track with a futuristic atmosphere, powerful basslines, deep punchy kick drums, crisp hi-hats, atmospheric synth pads, and catchy melodic arpeggios. Build tension with cinematic risers and dramatic transitions, leading into an explosive drop with heavy bass, energetic synth leads, and a driving rhythm. Use modern electronic sound design, polished professional production, wide stereo effects, and a festival-ready vibe. Tempo: 128 BPM. Instrumental only, no vocals. Make it immersive, energetic, euphoric, and perfect for a nightclub or music festival."

print("Cargando Musicgen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(text=[PROMPT],
                   padding=True,
               return_tensors="pt"
           ).to(DEVICE)

print("Generando musica...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1500
    )
sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0,0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Musica generada!")
