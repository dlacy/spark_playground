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

def readguide(guide):
    with open(guide, 'r') as myfile:
        return myfile.read()

def extract_text(content_str):
    content = json.loads(content_str)

    soup = BeautifulSoup(content[2], "lxml")

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
            return [content[0], content[1], txt.split(" ")]
        else:
            # if not a string, insert a space to make it one to avoid null content
            return [content[0], content[1], " "]
    else:
        # return empty string to avoid null content
        return [content[0], content[1], " "]

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

guides = glob("data/guides/*")

# create an RDD consisting of a list of paths to .json files
#guidesRDD = sc.parallelize(guides[0:10])
guidesRDD = sc.parallelize(guides)

# create an RDD containing html documents
documentsRDD = guidesRDD.map(lambda guide: readguide(guide))

# create RDD containg plain text derived from html, guide id, and page_id
textsRDD = documentsRDD.map(lambda document: extract_text(document))

textsDFrame = textsRDD.toDF(["guide_id", "page_id", "words"])

textsDFrame.show(10)

textsDFrame.printSchema()

# save to parquet
textsDFrame.select("guide_id", "page_id", "words").write.save("data/libguides_bow.parquet", format="parquet")
