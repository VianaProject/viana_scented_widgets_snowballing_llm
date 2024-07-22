# Scented widgets LLM code
### This repository is sharing the code which was used to call ChatGPT's and Gemini's API to assess whether a paper was including scented widgets or not.


## Set up
1. Install python 3.11.1
   - https://www.python.org/downloads/release/python-3111/
2. Install poetry (project was tested with version 1.4.2). This is a package manager which will install all required python libraries for the project to run.
   - https://python-poetry.org/docs/
   - in case newer version don't work the exact version of poetry can be located here:  https://github.com/python-poetry/poetry/releases/tag/1.4.2
3. Clone this project
4. In a terminal, change directory for this project's root
5. Install a virtual environment in the project's root
   - Windows: python -m venv .venv
   - Linux: python3 -m venv .venv
   * Naming the virtual environment .venv will help poetry to locate the virtual environment and install the required packages directly in this venv
6. Run `poetry install` to install all required packages into your newly created venv

## Running the project
#### There are 3 main scripts:
1. `chatGPTAPI.py`
2. `geminiTextAPI.py`
3. `geminiImageAPI.py`

The first script is using ChatGPT's API to assess whether a paper was including scented widgets or not. 

The second script is using Gemini's API to assess whether a paper was including scented widgets or not. 

The third script using Gemini's image processing capabilities to assess whether an image contains scented widgets or not.

These scripts lack the required TOKENS and/or Set up required for each of the providers.
In each script, at the beginning there is a link towards the documentation for each provider. 
The latter will help setting up your project further.

