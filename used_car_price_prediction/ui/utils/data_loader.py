import pandas as pd
import streamlit as st
from loguru import logger
from sqlalchemy import text

from scripts.dbmaker import SessionLocal

DATA_PATH = "./data/processed/autos_cleaned.csv"


@st.cache_data()
def load_data_from_csv():
    """Load and preprocess the dataset and visualization"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
        # return df.sample(30000)
    except FileNotFoundError:
        st.error("Dataset not found. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        st.error(f"An error occurred, cannot load data.")
        return pd.DataFrame()


# load data from database
@st.cache_data()
def load_data_from_db():
    try:
        db = SessionLocal()

        try:
            query = text(
                """
                SELECT brand, model, vehicle_type as vehicleType, fuel_type as fuelType,
                       power_ps as powerPS, kilometer, year_of_registration as yearOfRegistration,
                       gearbox, not_repaired_damage as notRepairedDamage, price
                FROM cars
                WHERE year_of_registration >= 2000
                LIMIT 10000
            """
            )
            result = db.execute(query).fetchall()
            df = pd.DataFrame(result)

            column_mapping = {
                "vehicle_type": "vehicleType",
                "fuel_type": "fuelType",
                "power_ps": "powerPS",
                "year_of_registration": "yearOfRegistration",
                "not_repaired_damage": "notRepairedDamage",
            }

            df.rename(columns=column_mapping, inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            st.error(f"Cannot load data from database.")

            df = load_data_from_csv()
            return df
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error connecting to the database: {str(e)}")
        st.error(f"Cannot connect to the database.")
        return pd.DataFrame()
