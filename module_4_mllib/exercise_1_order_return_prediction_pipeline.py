"""
Module 4 - MLlib
Exercise: Full Pipeline for order return prediction

Task:
1. Build a Pipeline encoding both category and customer_tier,
   assemble all features, use LogisticRegression as final estimator
2. Split 80/20, fit on train, transform both splits
3. Print predictions and probability on test set
4. Save, reload, and run inference on a new record
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

spark = SparkSession.builder.appName("Module4-Exercise1").getOrCreate()

orders = spark.createDataFrame([
    ("O001", "Electronics", "Premium",   299.99, 2, 1),
    ("O002", "Clothing",    "Standard",   45.00, 1, 0),
    ("O003", "Electronics", "Basic",     899.99, 5, 1),
    ("O004", "Home",        "Premium",   120.00, 1, 0),
    ("O005", "Clothing",    "Basic",      29.99, 3, 1),
    ("O006", "Electronics", "Standard",  199.99, 1, 0),
    ("O007", "Home",        "Standard",   89.99, 2, 0),
    ("O008", "Clothing",    "Premium",   150.00, 4, 1),
    ("O009", "Electronics", "Premium",   499.99, 1, 0),
    ("O010", "Home",        "Basic",      55.00, 3, 1),
], ["order_id", "category", "customer_tier", "order_amount", "prior_returns", "returned"])

# --- Feature Engineering Stages ---

# Stage 1: StringIndexer — encode both categoricals to numeric indices
category_indexer    = StringIndexer(inputCol="category",      outputCol="category_idx")
customer_tier_indexer = StringIndexer(inputCol="customer_tier", outputCol="tier_idx")

# Stage 2: OneHotEncoder — sparse binary vector per category
# Prevents the model treating index 2 as "twice" index 1
category_encoder = OneHotEncoder(inputCol="category_idx",  outputCol="category_ohe")
tier_encoder     = OneHotEncoder(inputCol="tier_idx",      outputCol="tier_ohe")

# Stage 3: VectorAssembler — combine all features into one vector
assembler = VectorAssembler(
    inputCols=["category_ohe", "tier_ohe", "order_amount", "prior_returns"],
    outputCol="raw_features"
)

# Stage 4: StandardScaler — zero mean, unit variance
# Important for LogisticRegression (gradient descent, distance-sensitive)
# Note: withMean=True forces dense vector conversion — on very wide OHE
# columns (hundreds of categories), consider withMean=False to preserve sparsity
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

# Stage 5: LogisticRegression (task specified LR, not GBT)
# regParam adds L2 regularization — helps prevent overfitting on small datasets
# NOTE: GBT (used in learner's solution) does NOT benefit from StandardScaler
# since tree-based models split on thresholds and are scale-invariant.
# LogisticRegression DOES benefit from scaling — good pairing here.
lr = LogisticRegression(
    labelCol="returned",
    featuresCol="features",
    maxIter=10,
    regParam=0.1
)

# --- Pipeline ---
# Fit once on train only — no data leakage from test set into
# StringIndexer mappings or StandardScaler mean/std
full_pipeline = Pipeline(stages=[
    category_indexer, customer_tier_indexer,
    category_encoder, tier_encoder,
    assembler, scaler, lr
])

train, test = orders.randomSplit([0.8, 0.2], seed=42)

# Cross-validation skipped: with only 10 rows, CV folds would have
# insufficient data diversity to produce meaningful hyperparameter estimates
model = full_pipeline.fit(train)

# Inspect learned parameters
trained_lr = model.stages[-1]
print("Coefficients:", trained_lr.coefficients)
print("Intercept:", trained_lr.intercept)

# Predict on test set
predictions = model.transform(test)
predictions.select(
    "order_id", "returned", "prediction", "probability"
).show(truncate=False)

# Evaluate
evaluator = BinaryClassificationEvaluator(labelCol="returned", metricName="areaUnderROC")
print(f"Test AUC: {evaluator.evaluate(predictions):.4f}")

acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="returned", predictionCol="prediction", metricName="accuracy"
)
print(f"Test Accuracy: {acc_evaluator.evaluate(predictions):.4f}")

# --- Save & Reload ---
model.write().overwrite().save("/tmp/returns_model")

loaded_model = PipelineModel.load("/tmp/returns_model")

# Explicit schema for new_order — production best practice
# avoids schema inference surprises (e.g. Long vs Int, Double vs Float)
schema = StructType([
    StructField("order_id",       StringType(), True),
    StructField("category",       StringType(), True),
    StructField("customer_tier",  StringType(), True),
    StructField("order_amount",   DoubleType(), True),
    StructField("prior_returns",  LongType(),   True),
    StructField("returned",       LongType(),   True),
])

new_order = spark.createDataFrame([
    ("O999", "Clothing", "Basic", 75.00, 2, None)
], schema)

loaded_model.transform(new_order) \
    .select("order_id", "category", "prediction", "probability") \
    .show(truncate=False)

# --- Key lessons from this exercise ---
# 1. Encode BOTH categoricals — missing one silently drops information
# 2. Fit pipeline on train ONLY — prevents data leakage into scaler/indexer
# 3. GBT doesn't need StandardScaler; LogisticRegression does
# 4. withMean=True forces dense vectors — watch memory on wide OHE columns
# 5. Explicit schema for inference inputs — avoids type mismatch in production
# 6. Identical probabilities across records = model collapsed features
#    (too little training data, not a code bug)
