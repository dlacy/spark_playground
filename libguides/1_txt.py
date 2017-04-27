import json
import re

from bs4 import BeautifulSoup

from glob import glob
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

def extract_text(content_str):

    if content_str is not None:
        soup = BeautifulSoup(content_str, "lxml")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        headerdiv = soup.find("div", {"id": "s-lg-guide-header-info"})
        maindiv = soup.find("div", {"id": "s-lg-guide-main"})

        if headerdiv and maindiv:
            txt = " ".join(headerdiv.strings).replace('\n', ' ') + " ".join(maindiv.strings).replace('\n', ' ')
            # remove non-alpha/numeric
            regex = re.compile('[^a-zA-Z]')
            txt = regex.sub(" ", txt)
            # remove whitespace
            txt = ' '.join(txt.split())
            if isinstance(txt, str):
                return txt
            else:
                return None

        else:
            return None
    else:
        return None


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Define udf
from pyspark.sql.functions import udf
udfextract_text=udf(extract_text, StringType())

raw = spark.read.load("data/all_guides.parquet")
raw.show(10)


print(" ---------- done load raw")

#textsDF = raw.limit(10).withColumn("words", udfextract_text(raw.content))
textsDF = raw.withColumn("words", udfextract_text(raw.content))

print(" ---------- done textDF")
textsDF.show(10)

print(" -------- textsDF.count(): ", textsDF.count())

#not_nulDF = textsDF.filter("words is not null")
#print(" -------- not_nulDF.count(): ", not_nulDF.count())


# save to parquet
textsDF.write.save("data/libguides_bow.parquet", format="parquet")
