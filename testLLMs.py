import torch
from transformer_lens import HookedTransformer

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Pythia-70M
print("Loading model...")
model = HookedTransformer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    device=device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)
model.eval()

# Input prompt
prompt = "Question: How many days did I spend on camping trips in the United States this year?"

# Convert to tokens
tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

# Generate output
print("Generating...")
generated_tokens = model.generate(
    tokens,
    max_new_tokens=50,
)

# Decode to text
output = model.to_string(generated_tokens[0])

print("\n=== MODEL OUTPUT ===")
print(output)
