from pyspark.sql import SparkSession, Row
import time
import xml.etree.ElementTree as ET
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import json
import ast

start = time.time()
time.clock()

spark = SparkSession.builder.master("local").getOrCreate()

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

es_write_conf = {
    "es.nodes": "localhost",
    "es.port": "9200",
    "es.resource": "cdm/records"
}

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


#records = df.select("record.*").where("record is not null").rdd.map(lambda row: ('key', udfparse_doc(row["document"])))
records = df.select("record.*").where("record is not null")
#records.show(10)
records2 = records.rdd.map(lambda row: ('key', parse_doc(row['document'])))

#ten = records2.take(10)

#print(ten)
#



#sets = df.select("set.*").where("record is not null")
#errors = df.select("error.*").where("error is not null")

#docs = records.withColumn("parsed_document", udfparse_doc(records.document))
#docs.show(10, False)

#save_docs = docs.select("parsed_document").rdd.map(lambda doc: ('key', ast.literal_eval(doc['parsed_document'])))

print("saving to ES...")

records2.saveAsNewAPIHadoopFile(
    path='-',
    outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
    keyClass="org.apache.hadoop.io.NullWritable",
    valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
    conf=es_write_conf)


#records.write.save("spark-warehouse/" + set + ".parquet", format="parquet")

elapsed = time.time() - start
print("seconds count: %02d" % elapsed)
#print("total records: %02d" % records.count())
