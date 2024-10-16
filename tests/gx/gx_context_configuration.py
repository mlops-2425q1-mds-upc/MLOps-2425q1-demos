import great_expectations as gx

from great_expectations.exceptions import DataContextError

# We import the existing DataContext
context = gx.get_context(mode="file")

# We load our training data and create a pandas datasource with it
datasource = context.data_sources.add_or_update_pandas(
    name="iowa_dataset",
)

data_asset = datasource.add_dataframe_asset(name="iowa_processed_data")
batch_definition = data_asset.add_batch_definition_whole_dataframe("iowa_training_data")

# We create an expectation suite for our training data
expectations_suite = gx.ExpectationSuite("iowa_training_data_validation")

try:
    context.suites.add(expectations_suite)
except DataContextError:
    expectations_suite = context.suites.get("iowa_training_data_validation")

expectations_suite.add_expectation(
    gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=[
            "MSSubClass",
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "OverallCond",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
            "MoSold",
            "YrSold",
            "Id",
            "SalePrice",
        ]
    )
)

expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="Id"))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Id"))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="Id", type_="int64"))

expectations_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(column="MSSubClass", min_value=0, max_value=200)
)
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="MSSubClass", type_="float64"))

expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="SalePrice"))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="SalePrice", min_value=0))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="SalePrice", type_="int64"))

# Only required if the suite already exists so that the changes are saved
expectations_suite.save()


# We create a validation definition to run our expectations suite
validator = gx.ValidationDefinition(
    data=batch_definition, suite=expectations_suite, name="iowa_training_data_validator"
)
try:
    context.validation_definitions.add(validator)
except DataContextError:
    context.validation_definitions.delete("iowa_training_data_validator")
    validator = context.validation_definitions.add(validator)

# We create a checkpoint to run our expectations and compile the results as Data Docs
action_list = [
    gx.checkpoint.UpdateDataDocsAction(name="update_data_docs"),
]
checkpoint = gx.Checkpoint(
    name="iowa_training_data_checkpoint",
    validation_definitions=[validator],
    actions=action_list,
    result_format={"result_format": "SUMMARY"},
)
try:
    context.checkpoints.add(checkpoint)
except DataContextError:
    context.checkpoints.delete("iowa_training_data_checkpoint")
    checkpoint = context.checkpoints.add(checkpoint)
