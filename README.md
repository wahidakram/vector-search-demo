# Vector DB - QA demo

## How to get started

1. Install a python virtual environment, e.g. `python3 -m venv venv`
2. Activate the virtual environment, e.g. `source venv/bin/activate`
3. Install the requirements: `pip install -r requirements.txt`
4. Make a copy of the `config.py.template` file and remove the `.template` suffix
5. Fill in the OpenAI API key and OpenAI organisation ID in the `config.py` file. This file is git-ignored to avoid sharing credentials.
6. Copy your PDF documents into the directory called `pdfs` in the root of this project
7. Run the script: `python main.py`