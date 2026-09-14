"""Loads .env once for the whole test session, so GEMINI_API_KEY /
ANTHROPIC_API_KEY set there (not just real shell environment variables)
reach os.environ before any test (especially tests/contract) needs them.
"""

from dotenv import load_dotenv

load_dotenv()
