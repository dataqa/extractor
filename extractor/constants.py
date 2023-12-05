from dotenv import load_dotenv
import os

load_dotenv()

AVERAGE_OUTPUT_TOKENS = 400
HOME = os.path.expanduser("~")
MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_COSTS = {"gpt-3.5-turbo": {"input": 0.0010, "output": 0.0020},
               "gpt-4": {"input": 0.03, "output": 0.06}}
