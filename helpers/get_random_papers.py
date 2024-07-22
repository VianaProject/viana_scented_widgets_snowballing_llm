import numpy as np
import pandas as pd
import random

# the year when this code was generated
random.seed(2024)

def get_10_percent_papers(df:pd.DataFrame, scented_widgets: str = "FALSE") -> None:
    subsetted_df = df[(df["scented_widgets_gemini_corrected"] == scented_widgets) & (df["scented_widgets_corrected"] == scented_widgets)]
    subsetted_df.reset_index(inplace=True, drop=True)
    number_of_papers = int(np.round(len(subsetted_df)*0.1))
    random_integers = [random.randint(0, len(subsetted_df) - 1) for _ in range(number_of_papers)]
    resubset = subsetted_df.iloc[random_integers]
    resubset.sort_values(by="index_gemini", inplace=True)
    resubset.reset_index(inplace=True, drop=True)
    print(resubset[["index_gemini", "Title"]])


def main():
    df = pd.read_csv('../records_complete_snowballing_chatgpt_and_gemini.csv')
    get_10_percent_papers(df)
    get_10_percent_papers(df, "TRUE")




if __name__ == '__main__':
    main()
