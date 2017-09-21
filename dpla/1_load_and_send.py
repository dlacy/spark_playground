from pyspark.sql import SparkSession, Row
import time
import xml.etree.ElementTree as ET
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import json
import ast

start = time.time()
time.clock()

spark = SparkSession.builder.appName("ingest_into_ES").getOrCreate()

records = spark.read.load("spark-warehouse/p15037coll3.parquet")

print("saving to ES...")

#this works, but slow
""" """
records = records.write\
    .format("org.elasticsearch.spark.sql")\
    .option("es.nodes", "45.33.93.64")\
    .option("es.port", "9200")\
    .option("es.index.auto.create", "true")\
    .option("es.nodes.wan.only", "true")\
    .option("es.resouce.auto.create", "cdm/test")\
    .mode("append")\
    .save("cdm/resource")

elapsed = time.time() - start
print("seconds count: %02d" % elapsed)
#print("total records: %02d" % records.count())
