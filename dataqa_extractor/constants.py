import os

HOME = os.path.expanduser("~")
MODEL = "gpt-3.5-turbo"
MODEL_COSTS = {"gpt-3.5-turbo": {"input": 0.0010, "output": 0.0020},
               "gpt-4": {"input": 0.03, "output": 0.06}}
