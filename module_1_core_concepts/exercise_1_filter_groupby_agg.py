"""
Module 1 - Core Concepts
Exercise: Filter + GroupBy + Aggregation on Server Logs

Task: Find total response_time_ms per server for ERROR events only,
      ordered by total response time descending.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum as spark_sum, col

spark = SparkSession.builder.appName("Module1-Exercise1").getOrCreate()

logs = spark.createDataFrame([
    ("web-01", "ERROR",   "2024-01-15 08:23:11", 512),
    ("web-02", "INFO",    "2024-01-15 08:24:05", 128),
    ("web-01", "ERROR",   "2024-01-15 08:25:30", 256),
    ("db-01",  "WARNING", "2024-01-15 08:26:00", 64),
    ("web-02", "ERROR",   "2024-01-15 08:27:15", 1024),
    ("db-01",  "ERROR",   "2024-01-15 08:28:45", 512),
    ("web-01", "INFO",    "2024-01-15 08:29:00", 256),
], ["server", "level", "timestamp", "response_time_ms"])

# --- Learner's solution (correct logic) ---
# response_time_ms = logs.filter(logs.level == "ERROR").groupBy("server").agg(sum("response_time_ms"))

# --- Polished solution ---
# Key improvements:
# 1. Use col() for filter condition (explicit column reference)
# 2. Alias sum as spark_sum to avoid shadowing Python's built-in sum()
# 3. Use .alias() to give the aggregated column a clean name
# 4. Add .orderBy() descending as required

error_response_times = (
    logs
    .filter(col("level") == "ERROR")
    .groupBy("server")
    .agg(spark_sum("response_time_ms").alias("total_response_time_ms"))
    .orderBy(col("total_response_time_ms").desc())
)

error_response_times.show()
# Expected output:
# +------+----------------------+
# |server|total_response_time_ms|
# +------+----------------------+
# |web-02|                  1024|
# |web-01|                   768|
# | db-01|                   512|
# +------+----------------------+
