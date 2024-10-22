import great_expectations as gx
import pandas as pd

from src.config import PROCESSED_DATA_DIR

# We import the existing DataContext
context = gx.get_context(mode="file")

checkpoint = context.checkpoints.get("iowa_training_data_checkpoint")

input_dir = PROCESSED_DATA_DIR / "iowa_dataset"
x_train = pd.read_csv(input_dir / "X_train.csv")
y_train = pd.read_csv(input_dir / "y_train.csv")

dataframe = pd.concat([x_train, y_train], axis=1)

batch_parameters = {"dataframe": dataframe}
checkpoint.run(batch_parameters=batch_parameters)
