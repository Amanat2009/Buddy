"""
Quick sanity-check for individual components (no hardware needed).
Run: python test_components.py
"""

print("Testing timer parsing...")
from context.timer import parse_timer_intent, extract_timer_label

cases = [
    ("set a timer for 20 minutes", 1200),
    ("remind me in an hour", 3600),
    ("give me 45 minutes", 2700),
    ("timer 5 mins", 300),
    ("wake me up in 30", 1800),
    ("30 second timer for pasta", 30),
    ("2 hours 30 minutes", 9000),
    ("what is the weather", None),
]

all_ok = True
for text, expected in cases:
    result = parse_timer_intent(text)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_ok = False
    print(f"  {status} '{text}' → {result}s (expected {expected}s)")

print()
print("Testing sentiment...")
from personality.sentiment import SentimentAnalyser
s = SentimentAnalyser()
s.load()
for text in ["I'm feeling great today!", "Everything is terrible.", "Just chilling."]:
    print(f"  '{text}' → {s.label(text)} ({s.score(text):.2f})")

print()
print("Testing system prompt builder...")
from llm.system_prompt import SystemPromptBuilder
from config import PERSONALITY_CORE, TIMEZONE
builder = SystemPromptBuilder(PERSONALITY_CORE, timezone=TIMEZONE)
prompt = builder.build()
print(f"  Prompt length: {len(prompt)} chars")
print(f"  Contains date: {'CURRENT DATE' in prompt}")

print()
print("Testing beep sounds (will play audio)...")
try:
    from audio.beep import play_wake_confirm
    play_wake_confirm()
    import time; time.sleep(0.5)
    print("  ✓ Beep OK")
except Exception as e:
    print(f"  ✗ Beep error: {e}")

print()
if all_ok:
    print("All tests passed!")
else:
    print("Some tests failed — check above.")
