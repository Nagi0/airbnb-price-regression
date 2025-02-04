import os
import pickle
import polars as pl
from ast import literal_eval
from dotenv import load_dotenv
from airbnbPriceRegression.humanAttempt.Modelling.models import Models
from airbnbPriceRegression.humanAttempt.DataLoading.data_loader import DataLoader
from airbnbPriceRegression.humanAttempt.Modelling.visualization import Visualization


if __name__ == "__main__":
    load_dotenv("airbnbPriceRegression/config/.env")
    VIEW_PLOTS = literal_eval(os.environ["VIEW_PLOTS"])

    dataset = DataLoader()
    dataset.preprocess(literal_eval(os.environ["DUMMY_COLUMNS"]))
    dataset = dataset.data

    if VIEW_PLOTS:
        Visualization().histogram(dataset, "price", p_title="Price Histogram")
        Visualization().histogram(dataset, "extra_people", p_title="Extra People Histogram")
        Visualization().boxplot(dataset, "price", p_title="Price Boxplot")
        Visualization().boxplot(dataset, "extra_people", p_title="Extra People Boxplot")

    dataset = dataset.filter(pl.col("price") <= pl.col("price").quantile(0.99))
    dataset = dataset.filter(pl.col("extra_people") <= pl.col("extra_people").quantile(0.99))

    if VIEW_PLOTS:
        Visualization().histogram(dataset, "price", p_title="Price Histogram")
        Visualization().histogram(dataset, "extra_people", p_title="Extra People Histogram")
        Visualization().boxplot(dataset, "price", p_title="Price Boxplot")
        Visualization().boxplot(dataset, "extra_people", p_title="Extra People Boxplot")

    model_type = "random_forest"
    modeling = Models(dataset, 0.20)
    # modeling.train_model(f"{model_type}")

    # save
    # with open(f"{model_type}.pkl", "wb") as f:
    #     pickle.dump(modeling.model, f)

    modeling.load_model(f"{model_type}.pkl")
    k_fold_results = modeling.cross_validation()
    print(k_fold_results)
