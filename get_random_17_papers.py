import pandas as pd
import random

# the year when this code was generated
random.seed(2024)


def main():
    df = pd.read_csv('records_complete_snowballing_chatgpt_and_gemini.csv')
    subsetted_df = df[(df["scented_widgets_gemini"] == "FALSE") & (df["scented_widgets"] == "FALSE")]
    subsetted_df.reset_index(inplace=True, drop=True)
    random_integers = [random.randint(0, len(subsetted_df)-1) for _ in range(17)]
    resubset = subsetted_df.iloc[random_integers]
    print(resubset["index"])




if __name__ == '__main__':
    main()
