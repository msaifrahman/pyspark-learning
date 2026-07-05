"""
Module 2 - Transformations & Queries
Exercise: Window functions (rank by total, month-over-month) + join strategy

Task:
1. Rank employees within each region by TOTAL sales across all months
2. Calculate month-over-month change in sales per employee
3. Join with a small regional_manager lookup table and identify join strategy
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, sum as spark_sum, rank, lag, broadcast

spark = SparkSession.builder.appName("Module2-Exercise1").getOrCreate()

sales_data = spark.createDataFrame([
    ("E001", "North", "2024-01", 15000.0),
    ("E001", "North", "2024-02", 18000.0),
    ("E001", "North", "2024-03", 12000.0),
    ("E002", "North", "2024-01", 22000.0),
    ("E002", "North", "2024-02", 19000.0),
    ("E002", "North", "2024-03", 25000.0),
    ("E003", "South", "2024-01",  9000.0),
    ("E003", "South", "2024-02", 11000.0),
    ("E003", "South", "2024-03", 14000.0),
], ["employee_id", "region", "month", "sales_amount"])

# --- Learner's solution (month-over-month logic correct; rank logic had a bug) ---
# window_spec = Window.partitionBy('employee_id').orderBy('month')
# result = sales_data.withColumn('total', spark_sum(col('sales_amount')).over(
#     window_spec.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
#     .withColumn('month_over_month', col("sales_amount") - lag(col("sales_amount"), 1).over(window_spec))
#
# Bug: rowsBetween(unboundedPreceding, currentRow) gives a RUNNING total per row,
# not the employee's grand total across all months. Also never called rank().
# Partition should be by "region" (not employee_id) to rank within region.

# --- Polished solution ---

# Part 1: Rank employees within region by TOTAL sales across all months
with_total = sales_data.withColumn(
    "employee_total",
    spark_sum("sales_amount").over(Window.partitionBy("region", "employee_id"))
)

ranked = with_total.withColumn(
    "rank_in_region",
    rank().over(Window.partitionBy("region").orderBy(col("employee_total").desc()))
)

ranked.select("employee_id", "region", "employee_total", "rank_in_region") \
    .distinct() \
    .orderBy("region", "rank_in_region") \
    .show()

# Part 2: Month-over-month change in sales per employee
mom_window = Window.partitionBy("employee_id").orderBy("month")

with_mom = sales_data.withColumn(
    "prev_month_sales", lag(col("sales_amount"), 1).over(mom_window)
).withColumn(
    "mom_change", col("sales_amount") - col("prev_month_sales")
)

with_mom.select("employee_id", "month", "sales_amount", "prev_month_sales", "mom_change") \
    .orderBy("employee_id", "month") \
    .show()

# Part 3: Join with regional manager lookup
regional_managers = spark.createDataFrame([
    ("North", "Priya Sharma"),
    ("South", "Daniel Osei"),
], ["region", "regional_manager"])

# This lookup table is tiny (a few KB) — well under the default 10MB
# autoBroadcastJoinThreshold. Spark will choose a BroadcastHashJoin:
# it copies regional_managers to every executor, avoiding a shuffle
# of the larger sales_data table entirely.
enriched = sales_data.join(broadcast(regional_managers), on="region")
enriched.explain()
# Look for "BroadcastHashJoin" / "BroadcastExchange" in the physical plan
enriched.show()
