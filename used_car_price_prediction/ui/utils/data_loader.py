import pandas as pd
import streamlit as st
from sqlalchemy import text

from scripts.dbmaker import SessionLocal

DATA_PATH = "./datasets/autos_cleaned.csv"


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
        st.error(f"An error occurred: {str(e)}")
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
                WHERE year_of_registration >= 2010
                LIMIT 1000
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

            # st.success(f"Loaded {len(df)} records from the database.")
            return df
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            # st.warning(
            #     "Failed to load data from the database. Loading from CSV instead."
            # )
            df = load_data_from_csv()
            return df
        finally:
            db.close()
    except Exception as e:
        st.error(f"Error connecting to the database: {str(e)}")
