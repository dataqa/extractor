import sys
from streamlit.web import cli as stcli


def run_app():
    sys.argv = ["streamlit", "run", "dataqa_extractor/extract.py"]
    sys.exit(stcli.main())


if __name__ == '__main__':
    run_app()
