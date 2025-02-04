import polars as pl
import matplotlib.pyplot as plt


class Visualization:

    @staticmethod
    def histogram(p_df: pl.DataFrame, p_column: str, p_title: str):
        plt.figure()
        plt.hist(p_df.select(f"{p_column}").to_numpy())
        plt.title(p_title)
        plt.show()

    @staticmethod
    def boxplot(p_df: pl.DataFrame, p_column: str, p_title: str):
        plt.figure()
        plt.boxplot(p_df.select(f"{p_column}").to_numpy())
        plt.title(p_title)
        plt.show()
