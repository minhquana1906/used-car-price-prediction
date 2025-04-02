import os
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sqlalchemy import text

from database.dbmaker import Base, Car, SessionLocal, engine, init_db

load_dotenv()


def convert_car_df_all(df):
    column_mapping = {
        "brand": "brand",
        "model": "model",
        "vehicleType": "vehicle_type",
        "fuelType": "fuel_type",
        "powerPS": "power_ps",
        "kilometer": "kilometer",
        "yearOfRegistration": "year_of_registration",
        "gearbox": "gearbox",
        "notRepairedDamage": "not_repaired_damage",
        "price": "price",
    }

    df_renamed = df.rename(columns=column_mapping)

    return df_renamed


# def convert_car_df(df):
#     column_mapping = {
#         "brand": "brand",
#         "model": "model",
#         "vehicleType": "vehicle_type",
#         "fuelType": "fuel_type",
#         "powerPS": "power_ps",
#         "kilometer": "kilometer",
#         "yearOfRegistration": "year_of_registration",
#         "gearbox": "gearbox",
#         "notRepairedDamage": "not_repaired_damage",
#         "price": "price",
#     }

#     # Apply stratified sampling to ensure diversity in the data
#     # First, create a combined feature for stratification based on important categorical features
#     logger.info(f"Original dataset size: {len(df)}")

#     # Create a stratification feature using brand and vehicle type
#     df["strat"] = df["brand"] + "_" + df["vehicleType"]

#     # Ensure we don't have too many strata with very few samples
#     strat_counts = df["strat"].value_counts()
#     min_samples = 5  # Minimum samples per stratum
#     valid_strata = strat_counts[strat_counts >= min_samples].index

#     # Filter to only include valid strata
#     df_valid = df[df["strat"].isin(valid_strata)]
#     logger.info(f"Valid data for stratified sampling: {len(df_valid)} rows")

#     n = 10000
#     # If we have less than 5000 valid samples, take all of them
#     if len(df_valid) <= n:
#         sample_df = df_valid
#         logger.info(f"Using all available {len(sample_df)} valid samples")
#     else:
#         # Calculate fraction to sample (minimum 1 from each stratum)
#         frac = min(1.0, n / len(df_valid))

#         try:
#             # Try stratified sampling
#             sample_df = df_valid.groupby("strat", group_keys=False).apply(
#                 lambda x: x.sample(frac=frac, random_state=42)
#             )

#             # If we have too many samples, take exactly n
#             if len(sample_df) > n:
#                 sample_df = sample_df.sample(n=n, random_state=42)

#             logger.info(f"Selected {len(sample_df)} samples using stratified sampling")

#         except Exception as e:
#             # Fallback to simple random sampling if stratified sampling fails
#             logger.warning(
#                 f"Stratified sampling failed: {e}. Using random sampling instead."
#             )
#             sample_df = df.sample(n=min(n, len(df)), random_state=42)
#             logger.info(f"Selected {len(sample_df)} samples using random sampling")

#     # Drop the stratification column
#     sample_df = sample_df.drop("strat", axis=1)

#     # Rename columns according to the mapping
#     df_renamed = sample_df.rename(columns=column_mapping)

#     # Log the distribution of key features
#     logger.info(
#         f"Distribution of brands: {df_renamed['brand'].value_counts().head(10)}"
#     )
#     logger.info(
#         f"Distribution of vehicle types: {df_renamed['vehicle_type'].value_counts()}"
#     )
#     logger.info(f"Distribution of fuel types: {df_renamed['fuel_type'].value_counts()}")
#     logger.info(
#         f"Distribution of gearbox types: {df_renamed['gearbox'].value_counts()}"
#     )
#     logger.info(
#         f"Distribution of damage status: {df_renamed['not_repaired_damage'].value_counts()}"
#     )

#     return df_renamed


def import_data_to_db(cfg):
    try:
        cleaned_data_path = cfg.paths.cleaned_data_file_path
        db_data = pd.read_csv(cleaned_data_path)
        db_data = convert_car_df_all(db_data)

        required_columns = [
            "brand",
            "model",
            "vehicle_type",
            "fuel_type",
            "power_ps",
            "kilometer",
            "year_of_registration",
            "gearbox",
            "not_repaired_damage",
            "price",
        ]

        missing_columns = [
            col for col in required_columns if col not in db_data.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return

        session = SessionLocal()

        try:
            car_count = session.query(Car).count()
            if car_count > 0:
                logger.info(
                    f"Car table already has {car_count} records. Skipping import."
                )
                return
            batch_size = 10000
            total_records = len(db_data)

            logger.info(
                f"Importing {total_records} car records in batches of {batch_size}"
            )

            for i in range(0, total_records, batch_size):
                batch = db_data.iloc[i : i + batch_size]

                # Create Car objects from DataFrame
                car_objects = []
                for _, row in batch.iterrows():
                    car = Car(
                        brand=row["brand"],
                        model=row["model"],
                        vehicle_type=row["vehicle_type"],
                        fuel_type=row["fuel_type"],
                        power_ps=float(row["power_ps"]),
                        kilometer=float(row["kilometer"]),
                        year_of_registration=int(row["year_of_registration"]),
                        gearbox=row["gearbox"],
                        not_repaired_damage=row["not_repaired_damage"],
                        price=float(row["price"]),
                    )
                    car_objects.append(car)

                session.add_all(car_objects)
                session.commit()

                logger.info(
                    f"Imported batch {i//batch_size + 1}/{(total_records-1)//batch_size + 1} ({i+len(batch)}/{total_records} records)"
                )

            logger.success(
                f"Successfully imported {total_records} car records into the database"
            )

        except Exception as e:
            session.rollback()
            logger.error(f"Error importing data: {str(e)}")
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    init_db()
    import_data_to_db(cfg)


if __name__ == "__main__":
    main()
