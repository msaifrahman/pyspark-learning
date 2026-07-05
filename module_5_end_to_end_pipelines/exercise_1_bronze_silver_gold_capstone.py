"""
Module 5 - End-to-End Pipelines
Capstone Exercise: Full Bronze → Silver → Gold pipeline with quarantine,
quality assertions, caching, and broadcast join

Task:
1. Bronze: add ingested_at and source metadata
2. Silver: cast types, deduplicate, quarantine bad records with rejection_reason,
   add quality assertion
3. Gold: user-level (purchases, refunds, net spend) and product-level
   (revenue, order count, avg price ordered by revenue desc) aggregates
4. Identify where to cache and where to broadcast join
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, lit, when, count,
    sum as spark_sum, avg, broadcast, try_to_timestamp
)
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("Module5-Capstone").getOrCreate()

raw_events = spark.createDataFrame([
    ("E001", "U001", "purchase", "laptop",    "1200.00", "2024-01-10 09:15:00"),
    ("E002", "U002", "purchase", "headphones",  "85.00", "2024-01-10 10:30:00"),
    ("E001", "U001", "purchase", "laptop",    "1200.00", "2024-01-10 09:15:00"),  # duplicate
    ("E003", "U001", "refund",   "laptop",    "1200.00", "2024-01-11 14:00:00"),
    ("E004", "U003", "purchase", "keyboard",   "45.00",  "2024-01-12 11:00:00"),
    ("E005", "U002", "purchase", "monitor",   "350.00",  "2024-01-13 16:45:00"),
    ("E006", "U004", "purchase", "laptop",   "1500.00",  "not-a-timestamp"),       # bad ts
    ("E007", "U003", "purchase", "mouse",      "25.00",  "2024-01-14 09:00:00"),
    ("E008", "U004", "purchase", "monitor",    None,     "2024-01-14 12:00:00"),   # null amount
    ("E009", "U005", "purchase", "headphones", "95.00",  "2024-01-15 08:30:00"),
], ["event_id", "user_id", "event_type", "product", "amount", "event_ts"])

# ── BRONZE: land raw data, add metadata only ──────────────────────────────
# Never filter or transform at this layer — it's your safety net
bronze = (
    raw_events
    .withColumn("ingested_at",      current_timestamp())
    .withColumn("source_file",      lit("events_20240115.csv"))
    .withColumn("pipeline_version", lit("1.0.0"))
)

# ── PRE-TYPING: cast once, reuse for both quarantine and silver ───────────
bronze_typed = bronze \
    .withColumn("amount_cast", col("amount").cast(DoubleType())) \
    .withColumn("ts_cast", try_to_timestamp(col("event_ts"), lit("yyyy-MM-dd HH:mm:ss")))

# ── QUARANTINE: capture bad records BEFORE any filtering ──────────────────
# Learner's solution correctly used try_to_timestamp (returns null on bad
# values instead of throwing). Also correctly ran quarantine before filtering.
# Gap: duplicates were silently dropped without logging — fixed below.

quarantine = (
    bronze_typed
    .filter(
        col("amount_cast").isNull() |
        (col("amount_cast") <= 0)  |
        col("ts_cast").isNull()
    )
    .withColumn("rejection_reason",
        when(col("amount_cast").isNull(), "null_amount")
        .when(col("amount_cast") <= 0,   "non_positive_amount")
        .when(col("ts_cast").isNull(),    "invalid_timestamp")
        .otherwise("unknown")
    )
    .withColumn("quarantined_at", current_timestamp())
)

# Log duplicate count separately so nothing is silently dropped
pre_dedup = (
    bronze_typed
    .filter(col("amount_cast").isNotNull())
    .filter(col("amount_cast") > 0)
    .filter(col("ts_cast").isNotNull())
)
dup_count = pre_dedup.count() - pre_dedup.dropDuplicates(["event_id"]).count()
print(f"Duplicates dropped:     {dup_count}")
print(f"Quarantined (bad data): {quarantine.count()}")
quarantine.select("event_id", "amount", "event_ts", "rejection_reason").show()

# ── SILVER: clean, typed, deduplicated ───────────────────────────────────
silver_valid = (
    bronze_typed
    .withColumn("amount",   col("amount_cast"))
    .withColumn("event_ts", col("ts_cast"))
    .drop("amount_cast", "ts_cast")
    .dropDuplicates(["event_id"])
    .filter(col("amount").isNotNull())
    .filter(col("amount") > 0)
    .filter(col("event_ts").isNotNull())
    .withColumn("cleaned_at", current_timestamp())
)

# Cache here, not at bronze — silver_valid is the expensive result
# (cast + dedup + filter) and feeds: 2x count(), 2x gold aggregations
# Without cache: Spark recomputes the full chain 4 times
# Learner cached bronze — correct instinct, wrong layer. Bronze is cheap
# (raw data + metadata). Silver is where the computation cost lives.
silver_valid.cache()
silver_valid.count()  # materialize the cache with one trigger action

actual_rows = silver_valid.count()
assert actual_rows >= 5, \
    f"Quality check FAILED: expected >=5 rows, got {actual_rows}"
print(f"✅ Quality check passed: {actual_rows} valid rows")
silver_valid.show(truncate=False)

# ── GOLD 1: user-level summary ────────────────────────────────────────────
# Separate purchases and refunds before aggregating — mixing them loses signal
# Learner's version grouped all event_types together, losing purchase/refund split
purchases = silver_valid.filter(col("event_type") == "purchase")
refunds   = silver_valid.filter(col("event_type") == "refund")

user_purchases = purchases.groupBy("user_id").agg(
    spark_sum("amount").alias("total_purchases"),
    count("event_id").alias("purchase_count")
)
user_refunds = refunds.groupBy("user_id").agg(
    spark_sum("amount").alias("total_refunds"),
    count("event_id").alias("refund_count")
)

# Left join so users with no refunds still appear in the output
user_summary = (
    user_purchases
    .join(user_refunds, on="user_id", how="left")
    .fillna(0, subset=["total_refunds", "refund_count"])
    .withColumn("net_spend", col("total_purchases") - col("total_refunds"))
    .orderBy(col("net_spend").desc())
)
user_summary.show()

# ── GOLD 2: product-level performance ────────────────────────────────────
# Learner grouped by (product, event_ts) — this fragments results into one
# row per product-per-timestamp instead of one aggregate row per product.
# Correct: group by product only, filter to purchases for revenue reporting.
product_performance = (
    purchases
    .groupBy("product")
    .agg(
        spark_sum("amount").alias("total_revenue"),
        count("event_id").alias("order_count"),
        avg("amount").alias("avg_price")
    )
    .orderBy(col("total_revenue").desc())
)
product_performance.show()

silver_valid.unpersist()  # release cache memory once gold is built

# ── BROADCAST JOIN: enrich with small product catalogue ───────────────────
# Learner correctly identified this as the right join strategy.
# product_catalogue is tiny (5 rows) — well under the 10MB broadcast
# threshold. Spark copies it to every executor, avoiding a shuffle of
# the larger silver_valid table entirely.
product_catalogue = spark.createDataFrame([
    ("laptop",      "Computers",   "TechBrand"),
    ("headphones",  "Audio",       "SoundCo"),
    ("keyboard",    "Accessories", "TechBrand"),
    ("monitor",     "Displays",    "ViewTech"),
    ("mouse",       "Accessories", "TechBrand"),
], ["product", "category", "brand"])

enriched = silver_valid.join(broadcast(product_catalogue), on="product")
enriched.select("event_id", "user_id", "product", "category", "brand", "amount").show()

# ── KEY LESSONS FROM THIS CAPSTONE ───────────────────────────────────────
# 1. Bronze = append-only landing zone, never filter or transform
# 2. Quarantine BEFORE filtering so every rejected record is accounted for
# 3. Log duplicates separately — dropDuplicates() is silent by default
# 4. Cache silver_valid (expensive result, reused 4x) not bronze (cheap)
# 5. Separate event_types before aggregating — don't mix purchases/refunds
# 6. groupBy on the right grain — groupBy("product", "event_ts") gives
#    one row per event, not one row per product
# 7. Broadcast small lookup/dimension tables to avoid shuffling large tables
# 8. unpersist() after gold is built to free cluster memory
