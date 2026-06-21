"""
Module 2 - Transformations & Queries
Exercise: String parsing without UDFs (split, regexp_extract) + when/otherwise

Task:
1. Extract transaction_type, category, and amount from a packed string
   WITHOUT using a UDF
2. Add net_effect: positive for PURCHASE, negative for REFUND
3. Thought exercise: when would split/regexp_extract NOT be sufficient,
   genuinely requiring a UDF?
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when

spark = SparkSession.builder.appName("Module2-Exercise2").getOrCreate()

transactions = spark.createDataFrame([
    ("T001", "C001", "PURCHASE-ELECTRONICS-299.99"),
    ("T002", "C002", "REFUND-CLOTHING-45.00"),
    ("T003", "C001", "PURCHASE-HOME-89.50"),
    ("T004", "C003", "PURCHASE-ELECTRONICS-1200.00"),
    ("T005", "C002", "REFUND-ELECTRONICS-299.99"),
], ["txn_id", "customer_id", "raw_description"])

# --- Learner's solution (correct logic, with one redundancy) ---
# Used regexp_extract on top of split() results — redundant since split()
# already isolates each clean piece (no extra text to filter out via regex).
# Also missing .otherwise() guard on net_effect when/otherwise chain.

# --- Polished solution ---

extracted = transactions.withColumn(
    "split_desc", split(col("raw_description"), "-")
).withColumn(
    "transaction_type", col("split_desc")[0]
).withColumn(
    "category", col("split_desc")[1]
).withColumn(
    "amount", col("split_desc")[2].cast("double")
).withColumn(
    "net_effect",
    when(col("transaction_type") == "PURCHASE", col("amount"))
    .when(col("transaction_type") == "REFUND", -col("amount"))
    .otherwise(None)  # guards against unexpected transaction types
).drop("split_desc")

extracted.show()

# --- Part 3: Thought exercise ---
# split()/regexp_extract() are sufficient when the format is a stable,
# clearly delimited pattern. A UDF becomes genuinely necessary when:
#
# 1. Variable structure - some records have 3 fields, others 5, with no
#    consistent pattern expressible via split or a single regex
# 2. Stateful parsing - meaning of field N depends on the value of field
#    N-1 (custom encoding schemes) - regex can't express conditional logic
# 3. External validation/lookup mid-parse - needing to look up a code
#    against a reference table WHILE parsing to decide how to interpret
#    the next token
# 4. Calling an external library - e.g. a dedicated address-parsing
#    library or NLP model with no Spark-native equivalent
#
# Rule of thumb: if parsing = "match this pattern" -> regex/split wins.
# If parsing requires branching on intermediate parsed values or external
# calls -> that's UDF (or Pandas UDF) territory.
