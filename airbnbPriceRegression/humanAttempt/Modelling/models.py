from dataclasses import dataclass
import keras
import polars as pl
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import tensorflow as tf
from keras import layers
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


@dataclass
class Models:
    dataset: pl.DataFrame
    test_size: float

    def __post_init__(self):
        dataset = self.dataset.to_pandas()
        dataset_index = dataset.index
        dataset_columns = dataset.columns

        self.scaler = StandardScaler()
        dataset = pd.DataFrame(self.scaler.fit_transform(dataset), index=dataset_index, columns=dataset_columns)
        self.dataset = pl.from_pandas(dataset)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            dataset.drop(columns=["price"]), dataset["price"], test_size=self.test_size, random_state=42
        )

    def load_model(self, p_path: str):
        if "h5" in p_path:
            model = keras.models.load_model(f"{p_path}")
            model.summary()

        else:
            with open(f"{p_path}", "rb") as f:
                model = pkl.load(f)

        self.model = model

    def evaluate_model(
        self, p_model: LinearRegression | RandomForestRegressor | Ridge | Lasso | MLPRegressor | keras.Sequential
    ):
        y_pred = p_model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print("Results:")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    def cross_validation(self, p_k_folds: int = 5):
        dataset = self.dataset.to_pandas()
        dataset_index = dataset.index
        dataset_columns = dataset.columns

        self.scaler = StandardScaler()
        dataset = pd.DataFrame(self.scaler.fit_transform(dataset), index=dataset_index, columns=dataset_columns)
        self.dataset = pl.from_pandas(dataset)

        results = []
        for idx in tqdm(range(p_k_folds)):
            _, X_test, _, y_test = train_test_split(
                dataset.drop(columns=["price"]), dataset["price"], test_size=self.test_size, random_state=idx
            )
            y_pred = self.model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({"mae": mae, "rmse": rmse, "r2": r2})

        return pd.DataFrame(results)

    def train_model(
        self,
        p_model: str,
        p_verbose: bool = True,
    ) -> LinearRegression | RandomForestRegressor | Ridge | Lasso | MLPRegressor | keras.Sequential:
        if p_model == "linear":
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)

        elif p_model == "random_forest":
            model = RandomForestRegressor(random_state=42, verbose=p_verbose)
            model.fit(self.X_train, self.y_train)

        elif p_model == "ridge":
            model = Ridge(random_state=42)
            model.fit(self.X_train, self.y_train)

        elif p_model == "lasso":
            model = Lasso(alpha=0.001, random_state=42)
            model.fit(self.X_train, self.y_train)

        elif p_model == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(10, 20, 100, 20, 10), activation="relu", solver="adam", verbose=p_verbose
            )
            model.fit(self.X_train, self.y_train)

        elif p_model == "tensorflow_mlp":
            gpus_list = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus_list:
                tf.config.experimental.set_memory_growth(gpu, True)

            model = keras.Sequential(
                [
                    layers.Dense(64, activation="relu", input_shape=[len(self.X_train.keys())]),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1),
                ]
            )
            callbacks = [
                ModelCheckpoint(
                    "tensorflow_mlp.{epoch:02d}-{val_loss:.4f}--0fold.h5",
                    verbose=p_verbose,
                    metric="val_loss",
                    save_best_only=True,
                ),
                EarlyStopping(patience=10, monitor="val_loss"),
                TensorBoard(
                    log_dir="C:\\GitHub\\airbnb-price-regression\\airbnbPriceRegression\\humanAttempt\\tensorflow_logs"
                ),
            ]
            model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
            model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=100,
                verbose=p_verbose,
                callbacks=callbacks,
            )

        self.evaluate_model(model)
        self.model = model
