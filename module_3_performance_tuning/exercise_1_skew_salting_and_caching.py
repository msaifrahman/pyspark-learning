"""
Module 3 - Performance Tuning & Optimization
Exercise: Skew detection + salting, correct caching, AQE limitations

Task:
1. Confirm skew exists in the sales dataset
2. Apply salting (15 buckets) to compute total revenue per region
3. Demonstrate caching done correctly: filter, cache, materialize, then
   run two aggregations against the cached result
4. Thought question: when is AQE's skewJoin.enabled insufficient?
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, concat_ws, lit, rand, split as spark_split,
    sum as spark_sum, avg
)
import time

spark = SparkSession.builder.appName("Module3-Exercise1").getOrCreate()

sales = spark.range(1_500_000).select(
    when(col("id") < 1_350_000, "REGION-WEST")
    .otherwise(concat_ws("-", lit("REGION"), (col("id") % 4).cast("string")))
    .alias("region"),
    (col("id") % 50).cast("string").alias("product_id"),
    (col("id") % 300 + 20).cast("double").alias("revenue")
)

# --- Part 1: Confirm the skew exists ---
sales.groupBy("region").count().orderBy(col("count").desc()).show()
# REGION-WEST will dwarf the other regions in row count

# --- Part 2: Salting technique (learner's solution - correct) ---
NUM_SALT_BUCKETS = 15

salted = sales.withColumn(
    "salted_key",
    concat_ws("_", col("region"), (rand() * NUM_SALT_BUCKETS).cast("int"))
)

# Stage 1: aggregate on salted keys - spreads the hot region's rows
# across 15 sub-partitions instead of 1
partial_agg = salted.groupBy("salted_key").agg(spark_sum("revenue").alias("partial_total"))

# Stage 2: strip the salt suffix and re-aggregate the (much smaller) partials
final_result = (
    partial_agg
    .withColumn("region", spark_split(col("salted_key"), "_")[0])
    .groupBy("region")
    .agg(spark_sum("partial_total").alias("total_revenue"))
)
final_result.orderBy(col("total_revenue").desc()).show(5)

# --- Part 3: Caching done correctly ---
# NOTE: original attempt cached a stray "orders" variable with no "revenue"
# column, and used revenue > 50 instead of the requested > 100.
# Corrected to use "sales" with the right filter threshold.

filtered_sales = sales.filter(col("revenue") > 100)

# .cache() is lazy - it marks the DataFrame for caching but doesn't
# materialize anything yet
filtered_sales.cache()

# This action triggers the actual computation AND writes the result into
# cache. Skipping this step means the next action does double work:
# compute the filter AND populate the cache at the same time.
filtered_sales.count()

# Both of these now read from the cached, already-filtered data -
# neither recomputes the filter from scratch
start = time.time()
print("Sum:", filtered_sales.agg(spark_sum("revenue")).collect())
print("Avg:", filtered_sales.agg(avg("revenue")).collect())
print(f"With cache: {time.time() - start:.2f}s")

filtered_sales.unpersist()  # release memory once done

# --- Part 4: Thought question ---
# AQE's skewJoin.enabled detects an oversized partition AFTER a shuffle
# (specific to JOIN operations) and splits it at runtime. It becomes
# insufficient when:
#
# 1. The skew is in a groupBy/aggregation, not a join - skewJoin.enabled
#    specifically targets join operations; pure aggregation skew (like
#    this exercise's region skew) isn't covered by it at all. This is
#    exactly why manual salting was required here - AQE's skew-join
#    handling would not have helped this aggregation scenario.
# 2. The skewed key has so much data that even after AQE splits it,
#    individual sub-partitions are still too large (e.g. one key with
#    95% of a 10-billion-row dataset).
# 3. The skew ratio doesn't cross AQE's detection threshold
#    (skewedPartitionFactor, default 5x median partition size) - if
#    skew is present but below this threshold, AQE won't trigger
#    splitting at all.
