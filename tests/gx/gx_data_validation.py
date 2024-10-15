import pandas as pd

import great_expectations as gx
from src.config import PROCESSED_DATA_DIR

# We import the existing DataContext
context = gx.get_context(mode="file")

checkpoint = context.checkpoints.get("iowa_training_data_checkpoint")

x_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv")

dataset = pd.concat([x_train, y_train], axis=1)

batch_parameters = {"dataset": dataset}
validation_result = checkpoint.run(batch_parameters=batch_parameters)
print(validation_result)
