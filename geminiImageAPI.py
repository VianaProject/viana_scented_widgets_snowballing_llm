from dataclasses import dataclass
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
import pandas as pd
import time
from helpers.preprocess_vis30k import create_image_objects
import logging


# create logger
logger = logging.getLogger('vis30k-logging')
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


@dataclass
class AnswerGemini:
    answer: str
    associated_paper_title: str
    paper_authors: str
    original_corpus_index: int
    widgets_in_paper: bool | str = None
    scented_widgets: bool | str = None

    def __post_init__(self):
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


def iterate_over_images():
    image_objs = create_image_objects()
    gem_answers = []
    for index, image_obj in enumerate(image_objs):
        # this is a manual way to restart where it stopped if a bug or a safety filter was triggered
        if index > -1: # put here the index to whichever index it stopped (to avoid additional costs).
            logger.info(f"processing image doi {image_obj.associated_doi} with the path {image_obj.path}")
            gemini_answer = generate_text_from_image(image_obj.path)
            image_obj.gemini_answer = gemini_answer
            gem_answers.append(AnswerGemini(answer=gemini_answer, associated_paper_title=image_obj.associated_doi, paper_authors=image_obj.pseudo_doi, original_corpus_index=index))
    logger.info(f"saving answers for index {index}")
    content = {
        "index_gemini": [answer.original_corpus_index for answer in gem_answers],
        "doi_gemini": [answer.associated_paper_title for answer in gem_answers],
        "title_gemini": [answer.paper_authors for answer in gem_answers],
        "widgets_in_paper_gemini": [answer.widgets_in_paper for answer in gem_answers],
        "scented_widgets_gemini": [answer.scented_widgets for answer in gem_answers],
        "answer_gemini": [answer.answer for answer in gem_answers],
    }
    df = pd.DataFrame(content)
    df.to_csv(f"vis30k_gem_answers{index}.csv", index=False)
    print("done")


def generate_text_from_image(image_path: str = "./test.png") -> str:
    # Load the model
    multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    # Query the model
    image = Image().load_from_file(image_path)
    # encoded_image = encode_image("./test.png")
    prompt = f"Does this image show any user interface widgets such as sliders, buttons, lists etc.? Answer only by 'yes', 'no' or 'I don't know'. If yes, are these widgets scented (or enhanced, sophisticated) in the visual analytics sense?. Answer again only by with yes, no or I don't know."
    response = multimodal_model.generate_content(
        [
            # Add an example image
            Part.from_image(image),
            # Add an example query
            prompt,
        ]
    )
    try:
        logger.info(response.text)
        return response.text
    except ValueError:
        logger.info("blocked by safety filters or quota reached...")
        return "blocked by safety filters or quota reached..."


def main():
    iterate_over_images()



if __name__ == '__main__':
    main()
