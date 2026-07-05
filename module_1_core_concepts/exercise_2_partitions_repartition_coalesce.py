"""
Module 1 - Core Concepts
Exercise: Partitions — repartition vs coalesce

Task:
1. Check default number of partitions
2. Repartition by region_code into 10 partitions
3. Verify row distribution across partitions
4. Coalesce to 5 partitions for output
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, spark_partition_id, count

spark = SparkSession.builder.appName("Module1-Exercise2").getOrCreate()

transactions = spark.range(500_000).select(
    col("id").alias("txn_id"),
    (col("id") % 1000).cast("string").alias("customer_id"),
    (col("id") % 10).alias("region_code"),
    (col("id") % 200 + 1).cast("double").alias("amount")
)

# --- Learner's solution (correct logic, minor placement issue with partition_id) ---
# region_transactions = transactions.repartition(10, "region_code")
# region_transactions.groupBy("region_code").count()
#     .withColumn("partition", spark_partition_id()).orderBy("region_code").show()
# Note: spark_partition_id() after groupBy shows where aggregated rows landed,
#       not the partition distribution of source rows.

# --- Polished solution ---

# Step 1: Check default partitions before any transformation
print("Default partitions:", transactions.rdd.getNumPartitions())

# Step 2: Repartition by region_code — co-locates all rows of the same region
# into the same partition, which is efficient for downstream groupBy on region_code
region_transactions = transactions.repartition(10, col("region_code"))
print("After repartition:", region_transactions.rdd.getNumPartitions())  # 10

# Step 3: Verify distribution — apply spark_partition_id() BEFORE groupBy
# so we see how source rows are spread, not where aggregation results landed
region_transactions \
    .withColumn("partition", spark_partition_id()) \
    .groupBy("partition", "region_code") \
    .agg(count("*").alias("row_count")) \
    .orderBy("partition") \
    .show(20)

# Step 4: Coalesce to 5 for output
# Using coalesce (not repartition) because:
# - We are only reducing partition count, not redistributing data
# - coalesce merges existing partitions locally — no shuffle across the network
# - Much cheaper than repartition when decreasing partition count
# - Produces fewer output files, which speeds up downstream reads
transaction_output = region_transactions.coalesce(5)
print("After coalesce:", transaction_output.rdd.getNumPartitions())  # 5
