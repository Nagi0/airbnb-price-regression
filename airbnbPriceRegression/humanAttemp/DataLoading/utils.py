import polars as pl


class PolarsUtils:

    @staticmethod
    def data_clean(p_df: pl.DataFrame):
        return p_df.lazy().drop_nulls().drop_nans().collect()

    @staticmethod
    def column_to_datetime(p_df: pl.DataFrame, p_column: str):
        return p_df.lazy().with_columns(pl.col(p_column).str.to_datetime("%Y-%m-%d")).collect()

    @staticmethod
    def replace_value(p_df: pl.DataFrame, p_column: str, p_old_value, p_new_value):
        return p_df.lazy().with_columns(pl.col(p_column).replace(p_old_value, p_new_value)).collect()

    @staticmethod
    def count_elements(p_df: pl.DataFrame, p_column: str, p_sep: str, p_alias: str):
        return p_df.with_columns(
            pl.col(p_column).str.split(p_sep).map_elements(lambda x: len(x), return_dtype=pl.Int32).alias(p_alias)
        )

    @staticmethod
    def price_to_float(p_df: pl.DataFrame, p_column: str):
        return (
            p_df.lazy()
            .with_columns(
                pl.col(p_column).map_elements(
                    lambda x: float(x.replace("$", "").replace(",", "")), return_dtype=pl.Float32
                )
            )
            .collect()
        )
