import numpy as np
import pandas as pd
import random
from Levenshtein import distance as lev

# the year when this code was generated
random.seed(2024)

def get_randomn_papers_for_text_llms(df:pd.DataFrame, scented_widgets: str = "FALSE") -> None:
    subsetted_df = df[(df["scented_widgets_gemini_corrected"] == scented_widgets) & (df["scented_widgets_corrected"] == scented_widgets)]
    subsetted_df.reset_index(inplace=True, drop=True)
    number_of_papers = int(np.round(len(subsetted_df)*0.1))
    random_integers = [random.randint(0, len(subsetted_df) - 1) for _ in range(number_of_papers)]
    resubset = subsetted_df.iloc[random_integers]
    resubset.sort_values(by="index_gemini", inplace=True)
    resubset.reset_index(inplace=True, drop=True)
    print(resubset[["index_gemini", "Title"]])


def get_randomn_papers_for_image_parsing_ai(df:pd.DataFrame, scented_widgets: str = "FALSE") -> None:
    number_of_random_papers = 20
    subsetted_df = df[df["scented_widgets_gemini_corrected"] == scented_widgets]
    subsetted_df.reset_index(inplace=True, drop=True)
    random_integers = [random.randint(0, len(subsetted_df) - 1) for _ in range(number_of_random_papers)]
    resubset = subsetted_df.iloc[random_integers]
    resubset.sort_values(by="index_gemini", inplace=True)
    print(resubset[["index_gemini", "doi_gemini"]])



def main():
    # that's for the text parser
    df = pd.read_csv('../data/records_complete_snowballing_chatgpt_and_gemini.csv')
    get_randomn_papers_for_text_llms(df)
    get_randomn_papers_for_text_llms(df, "TRUE")
    # that's for the image parser
    df_image =  pd.read_csv('../data/vis30k_gemini_unique_answers.csv')
    get_randomn_papers_for_image_parsing_ai(df_image)
    get_randomn_papers_for_image_parsing_ai(df_image, "TRUE")




if __name__ == '__main__':
    main()
