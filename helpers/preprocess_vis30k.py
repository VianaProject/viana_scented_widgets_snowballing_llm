import os
from dataclasses import dataclass
import pathlib
import pandas as pd
import logging

# create logger
logger = logging.getLogger('vis30k-preprocessing')
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

# here we configure vis30K images location. As it's a very heavy it will not be added to this project.
# feel free to download it locally and go from there.
images_path = pathlib.Path(__file__).parent.parent.absolute() / "images"
metadata_path = pathlib.Path(__file__).parent.parent.absolute() / "metadata"


@dataclass
class Vis30kImage:
    path: str | pathlib.Path
    year: int
    pseudo_doi: str
    associated_doi: str
    gemini_answer: str = ""
    has_widgets: bool = False
    has_scented_widgets: bool = False


def iterate_through_years(path: str | pathlib.Path) -> [Vis30kImage]:
   years = os.listdir(path)
   image_objs = []
   for year in years:
       logger.info("Processing year: " + year)
       for image in os.listdir(path / year):
           logger.info("Processing image: " + image)
           doi_pseudo = get_doi_from_year_and_image(year, image)
           doi = _format_doi(doi_pseudo)
           temp_im = Vis30kImage(path / year / image, int(year), doi_pseudo, doi)
           image_objs.append(temp_im)
   return image_objs


def get_doi_from_year_and_image(year: str, image: str) -> str:
    meta = os.listdir(metadata_path)
    right_meta = [m for m in meta if m.__contains__(year)][0]
    df = pd.read_csv(metadata_path / right_meta, header=None)
    doi_col = df.iloc[2:,1].to_list()
    image_page_number_number = image.split(".")[1]
    image_vis_conf_name = image.split(".")[0][:-1]
    right_doi = [doi for doi in doi_col if _is_this_the_right_doi(doi, image_page_number_number, image_vis_conf_name)][0]
    return right_doi


def _format_doi(pseudo_doi: str) -> str:
    parts = pseudo_doi.split("-")
    parts = parts[1:]
    new_parts_joined = "-".join(parts)
    almost_doi = new_parts_joined.split("-p")[0]
    link_doi = "https://doi.org/" + almost_doi.replace("-", "/")
    return link_doi


def _is_this_the_right_doi(identifier: str, image_page_number_number: int, image_vis_conf_name: str) -> bool:
    identifier_parts = identifier.split("-")
    conf = identifier_parts[0].replace(" ", "")
    page_number = [part for part in identifier_parts if __is_this_a_page(part)][0][1:]
    if conf.lower() == image_vis_conf_name.lower() and int(page_number) == int(image_page_number_number):
        return True
    else:
        return False


def __is_this_a_page(part:str) -> bool:
    if part[0] == "p":
        try:
            int(part[1])
            return True
        except ValueError:
            return False
    return False


def create_image_objects() -> [Vis30kImage]:
    objs = iterate_through_years(images_path)
    return objs


if __name__ == '__main__':
    objs = create_image_objects()

