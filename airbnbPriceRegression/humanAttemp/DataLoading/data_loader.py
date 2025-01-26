import os
from ast import literal_eval
from dataclasses import dataclass
from dotenv import load_dotenv
import polars as pl
from kagglehub import dataset_download
from airbnbPriceRegression.humanAttemp.DataLoading.utils import PolarsUtils


@dataclass()
class DataLoader:
    data: pl.DataFrame = pl.DataFrame

    def __post_init__(self):
        load_dotenv("./airbnbPriceRegression/config/.env")
        path = dataset_download(os.environ["DATA_PATH"])
        selected_columns_list = literal_eval(os.environ["SELECTED_COLUMNS"])
        self.data = pl.read_csv(f"{path}/total_data.csv").select(selected_columns_list)

    def data_clean(self):
        self.data = PolarsUtils.data_clean(self.data)

    def feature_eng(self):
        self.data = PolarsUtils.column_to_datetime(self.data, "host_since")
        most_recent_year = self.data.select(pl.col("host_since").max()).to_pandas()["host_since"][0].year
        self.data = (
            self.data.lazy()
            .with_columns((most_recent_year - pl.col("host_since").dt.year()).alias("host_since_years"))
            .collect()
        ).drop("host_since")

        self.data = PolarsUtils.replace_value(self.data, "host_is_superhost", "f", False)
        self.data = PolarsUtils.replace_value(self.data, "host_is_superhost", "t", True)

        self.data = PolarsUtils.replace_value(self.data, "host_identity_verified", "f", False)
        self.data = PolarsUtils.replace_value(self.data, "host_identity_verified", "t", True)

        self.data = PolarsUtils.replace_value(self.data, "is_location_exact", "f", False)
        self.data = PolarsUtils.replace_value(self.data, "is_location_exact", "t", True)

        self.data = PolarsUtils.replace_value(self.data, "instant_bookable", "f", False)
        self.data = PolarsUtils.replace_value(self.data, "instant_bookable", "t", True)

        self.data = PolarsUtils.replace_value(self.data, "is_business_travel_ready", "f", False)
        self.data = PolarsUtils.replace_value(self.data, "is_business_travel_ready", "t", True)

        self.data = PolarsUtils.count_elements(self.data, "host_verifications", ",", "host_verifications_number").drop(
            "host_verifications"
        )
        self.data = PolarsUtils.count_elements(self.data, "amenities", ",", "amenities_number").drop("amenities")

        self.data = PolarsUtils.price_to_float(self.data, "price")
        self.data = PolarsUtils.price_to_float(self.data, "extra_people")

    def create_dummies(self, p_dummy_columns: list):
        self.data = self.data.to_dummies(columns=p_dummy_columns)

    def preprocess(self, p_dummy_columns: list):
        self.data_clean()
        self.feature_eng()
        self.create_dummies(p_dummy_columns)


if __name__ == "__main__":
    data_loader = DataLoader()
    dummy_columns = literal_eval(os.environ["DUMMY_COLUMNS"])
    data_loader.preprocess(dummy_columns)
    print(data_loader)
