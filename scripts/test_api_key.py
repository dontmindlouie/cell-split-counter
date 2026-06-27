"""Quick smoke test — run this after setting ANTHROPIC_API_KEY to verify the setup."""

import os
from pathlib import Path
import anthropic

# Load .env if present (avoids needing setx before first run)
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    print("ERROR: ANTHROPIC_API_KEY is not set.")
    raise SystemExit(1)

print("Key found.")

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=32,
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
)
print(f"API response: {response.content[0].text.strip()}")
print("Setup complete.")
