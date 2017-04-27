import json
import urllib
import urllib.request
from http.cookiejar import CookieJar
import requests
import html
import re
import time
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType


def get_guide(url):

    import requests

    try:
        response = requests.get(url, allow_redirects=False)

        if response.status_code == 200:
            # Try to get the data and json.load it 5 times, then give up
            tries = 5
            while tries >= 0:
                try:
                    print("  ---------- trying: ", url)
                    content = response.text
                    return content
                except:
                    if tries == 0:
                        # If we keep failing, raise the exception for the outer exception
                        # handling to deal with
                        raise
                    else:
                        # Wait a few seconds before retrying and hope the problem goes away
                        time.sleep(3)
                        tries -= 1
                        continue
    except:
        raise

spark = SparkSession.builder.appName("harvester").getOrCreate()

guides = spark.read.json('data/flattened_guides.json')

guides.show(10)

guides.printSchema()

# Define udf
from pyspark.sql.functions import udf
udfget_guide=udf(get_guide, StringType())

guidesDF = guides.withColumn("content", udfget_guide(guides.url))

print("done")

# save to parquet
guidesDF_filtered = guidesDF.rdd.filter(lambda x: x is not None)

guidesDF = guidesDF_filtered.toDF()

guidesDF.show(10)

guidesDF.write.save("data/all_guides.parquet", format="parquet")
