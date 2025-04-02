from datetime import datetime

import pandas as pd

# df = pd.read_csv("./datasets/cleaned_dataset.csv")
# print(df.columns)

# df["yearOfRegistration"] = datetime.now().year - df["age"]
# df = df.drop(columns=["age", "Unnamed: 0"], axis=1)
# df["gearbox"] = df["gearbox"].map({1: "automatic", 0: "manual"})
# df["notRepairedDamage"] = df["notRepairedDamage"].map({1: "yes", 0: "no"})

# df.to_csv("./datasets/autos_cleaned.csv", index=False)


df = pd.read_csv("./datasets/autos_cleaned.csv")

print(df[df["yearOfRegistration"] >= 2010].count())
