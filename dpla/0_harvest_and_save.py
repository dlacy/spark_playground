from pyspark.sql import SparkSession, Row
import time
import xml.etree.ElementTree as ET
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import json
import ast

start = time.time()
time.clock()

spark = SparkSession.builder.appName("harvest_CDM").getOrCreate()

ns = {'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/',
      'dc': 'http://purl.org/dc/elements/1.1/'}

def parse_doc(document_str):
    global ns
    dc = {}
    root = ET.fromstring(document_str)
    oai_dc = root.findall('.//oai_dc:dc/*', ns)
    for child in oai_dc:
        tag = child.tag
        #print(tag)
        # remove the ns URI and use property as dict key
        dc[tag.split("}")[1]] = child.text
    return dc
    #return json.dumps(dc)

# 50,000+ p15037coll3
# 4,000+ p16002coll9

set = "p15037coll3"

df = spark.read.\
    format("dpla.ingestion3.harvesters.oai")\
    .option("endpoint", "http://digital.library.temple.edu/oai/oai.php")\
    .option("verb", "ListRecords")\
    .option("setlist", set)\
    .option("metadataPrefix", "oai_dc")\
    .load()

# Define udf
from pyspark.sql.functions import udf
udfparse_doc = udf(parse_doc, StringType())

records = df.select("record.*").where("record is not null")\
    .rdd\
    .map(lambda row: (parse_doc(row["document"])))\
    .toDF()\
    .write\
    .save("spark-warehouse/" + set + ".parquet", format="parquet")

#
print("records.rdd.getNumPartitions():")
print(records.rdd.getNumPartitions())

elapsed = time.time() - start
print("seconds count: %02d" % elapsed)
#print("total records: %02d" % records.count())
