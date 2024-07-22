from dataclasses import dataclass
from openai import OpenAI
import os
import pandas as pd

# see more details how to set up your ChatGPT project here:
# You'll need a certificate and its path added to your environment variables as SSL_CERT_FILE
# and a OPENAI_API_KEY also added to your environment variables

os.environ["SSL_CERT_FILE"] = 'path_to_your .pem certificate goes here. ' \
                              'See more details here:  https://platform.openai.com/docs/api-reference/introduction'
client = OpenAI()
snowballing_corpus = pd.read_csv('data/records_complete_snowballing.csv')


def ask_text_to_chatgpt(index: int, title: str, authors: str):
    base_message = f"Does the paper '{title}' by {authors} show in their figures any user interface widgets such as sliders, buttons, lists etc.? Answer only by 'yes', 'no' or 'I don't know'. If yes, are these widgets scented (or enhanced, sophisticated) in the visual analytics sense?. Answer again only by with yes, no or I don't know."
    completion = client.chat.completions.create(
        # model="gpt-4-vision-preview",
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system",
             "content": "You are a researcher in visual analytics."},
            {"role": "user", "content": base_message}
        ]
    )
    print(completion.choices[0].message)
    answer_chat_gpt = AnswerChatGPT(completion.usage.total_tokens, completion.choices[0].message.content, title, authors, index)
    return answer_chat_gpt


@dataclass
class AnswerChatGPT:
    spend_tokens: int
    answer: str
    associated_paper_title: str
    paper_authors: str
    original_corpus_index: int
    widgets_in_paper: bool | str = None
    scented_widgets: bool | str = None

    def __post_init__(self):
        if self.answer.lower().__contains__("yes"):
            self.widgets_in_paper = True
            if len(self.answer.lower().split("yes")) > 2:
                self.scented_widgets = True
            else:
                self.scented_widgets = False
        elif self.answer.lower().__contains__("i don't know"):
            self.widgets_in_paper = "Unknown"
            self.scented_widgets = "Unknown"
        elif self.answer.lower().__contains__("no"):
            self.widgets_in_paper = False
            self.scented_widgets = False
        else:
            self.widgets_in_paper = "Unknown"
            self.scented_widgets = "Unknown"


def iterate_over_corpus(corpus: pd.DataFrame) -> [AnswerChatGPT]:
    answers = [];
    for index, row in corpus.iterrows():
        answer = ask_text_to_chatgpt(index, row['Title'], row['Authors'])
        answers.append(answer)
    return answers


def concatenate_with_corpus(answers: [AnswerChatGPT]) -> pd.DataFrame:
    content = {
        "index": [answer.original_corpus_index for answer in answers],
        "title": [answer.associated_paper_title for answer in answers],
        "authors": [answer.paper_authors for answer in answers],
        "widgets_in_paper": [answer.widgets_in_paper for answer in answers],
        "scented_widgets": [answer.scented_widgets for answer in answers],
        "answer": [answer.answer for answer in answers],
        "spend_tokens": [answer.spend_tokens for answer in answers]
    }
    df = pd.DataFrame(content)
    concatenated_df = pd.concat([df, snowballing_corpus], axis=1)
    return concatenated_df


def main():
    answers = iterate_over_corpus(snowballing_corpus)
    concatenated_df = concatenate_with_corpus(answers)
    # our results file is already located at that location
    concatenated_df.to_csv('data/records_complete_snowballing_chatgpt.csv', index=False)


if __name__ == '__main__':
    main()
