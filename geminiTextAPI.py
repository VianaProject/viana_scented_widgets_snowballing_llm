from dataclasses import dataclass
import vertexai
from vertexai.generative_models import GenerativeModel, Part, ChatSession, Image
import pandas as pd
import time
import unicodedata
import logging


# create logger
logger = logging.getLogger('logging Gemini requests')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
# ch = logging.FileHandler('vis30k-logging.log')
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


# set up your project with Gemini and VertexAI.
# documentation available here: https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal
# there is a need to install google's console and config your project locally with your credentials for this code to work.
# follow the above documentation's guidelines.
vertexai.init(project="use your project id", location="use your location but bear in mind, API speed and limits are location dependent")
model = GenerativeModel("gemini-1.0-pro-001")
# I have run ChatGPT before GEMINI so here it assumes the same.
# The script runs from there accordingly even if there is no coupling. You could use the same starting point as we did
# for ChatGPT
corpus = pd.read_csv("data/records_complete_snowballing_chatgpt.csv")


@dataclass
class AnswerGemini:
    answer: str
    associated_paper_title: str
    paper_authors: str
    original_corpus_index: int
    widgets_in_paper: bool | str = None
    scented_widgets: bool | str = None

    def __post_init__(self):
        # todo here some post processing of the answers. Needs to be improved. Is not accurate enough.
        time.sleep(0.2)
        if self.answer.lower().__contains__("yes"):
            self.widgets_in_paper = True
            if len(self.answer.lower().split("yes")) > 2:
                self.scented_widgets = True
            elif self.answer.lower().__contains__("i don't know"):
                self.scented_widgets = "Unknown"
            else:
                self.scented_widgets = False
        elif self.answer.lower().__contains__("no") and (not self.answer.lower().__contains__("i don't know")):
            self.widgets_in_paper = False
            self.scented_widgets = False
        elif self.answer.lower().__contains__("i don't know"):
            temp = self.answer.lower().split("i don't know")
            d = [i.__contains__("no") for i in temp]
            if any(d):
                self.widgets_in_paper = False
                self.scented_widgets = False
            else:
                self.widgets_in_paper = "Unknown"
                self.scented_widgets = "Unknown"
        elif self.answer.lower().__contains__("no"):
            self.widgets_in_paper = False
            self.scented_widgets = False
        else:
            self.widgets_in_paper = "Unknown"
            self.scented_widgets = "Unknown"


def iterate_over_corpus(corpus: pd.DataFrame) -> [AnswerGemini]:
    answers = [];
    for index, row in corpus.iterrows():
        print(f"preparing to ask for answer for index {index}, paper: " + str(row['Title']) + " by " + str(row['Authors']))
        answer = ask_gemini(index, row['Title'], row['Authors'])
        answers.append(answer)
    return answers


def concatenate_with_corpus(answers: [AnswerGemini]) -> pd.DataFrame:
    content = {
        "index_gemini": [answer.original_corpus_index for answer in answers],
        "title_gemini": [answer.associated_paper_title for answer in answers],
        "authors_gemini": [answer.paper_authors for answer in answers],
        "widgets_in_paper_gemini": [answer.widgets_in_paper for answer in answers],
        "scented_widgets_gemini": [answer.scented_widgets for answer in answers],
        "answer_gemini": [answer.answer for answer in answers],
    }
    df = pd.DataFrame(content)
    concatenated_df = pd.concat([df, corpus], axis=1)
    return concatenated_df


def convert_to_ascii(input_str):
    # Normalize the input string to decompose the special characters
    normalized_str = unicodedata.normalize('NFKD', input_str)
    # Filter out the non-ASCII characters
    ascii_str = ''.join(c for c in normalized_str if ord(c) < 128)
    return ascii_str


def ask_gemini(index: int, title: str, authors: str) -> AnswerGemini:
    prompt = f"Does the paper {title} by {authors} show in their figures any user interface widgets such as sliders, buttons, lists etc.? Answer only by 'yes', 'no' or 'I don't know'. If yes, are these widgets scented (or enhanced, sophisticated) in the visual analytics sense?. Answer again only by with yes, no or I don't know."
    def get_chat_response(chat: ChatSession, prompt: str) -> str:
        text_response = []
        responses = chat.send_message(prompt, stream=True)
        for chunk in responses:
            text_response.append(chunk.text)
        return "".join(text_response)
    chat = model.start_chat()
    answer = get_chat_response(chat, prompt)
    print(answer)
    answer_gemini = AnswerGemini(answer=answer, associated_paper_title=title,
                                 paper_authors=authors, original_corpus_index=index)
    return answer_gemini


def main():
    answers_gemini = iterate_over_corpus(corpus)
    concatenated_df = concatenate_with_corpus(answers_gemini)
    concatenated_df.to_csv('records_complete_snowballing_chatgpt_and_gemini.csv', index=False)


if __name__ == '__main__':
    main()
