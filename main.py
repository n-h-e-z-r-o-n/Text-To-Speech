# Install first:
# pip install chatterbox-tts

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)

model = ChatterboxTurboTTS.from_pretrained(
    device=device
)

text = """
Hmm... okay.

I wasn't expecting that. [chuckle]
But you know what? Maybe that's actually a good thing.

[laugh]

Alright, let's try this again.
"""

wav = model.generate(text)

ta.save(
    "chatterbox_test.wav",
    wav,
    model.sr
)

print("Saved: chatterbox_test.wav")