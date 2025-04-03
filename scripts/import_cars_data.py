import os
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sqlalchemy import text

from scripts.dbmaker import Base, Car, SessionLocal, engine, init_db

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
