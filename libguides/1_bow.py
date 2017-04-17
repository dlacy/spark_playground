#source activate py35
#/Users/dlacy/dev/machine_learning_tutorial/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --master local[6] /Users/dlacy/dev/BuildingMachineLearningSystemsWithPython/ch03/create_LG_dict_via_spark.py

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.externals import joblib
#from pyspark.ml.feature import HashingTF, IDF, Tokenizer
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

def extract_text(content):
    soup = BeautifulSoup(content, "lxml")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    headerdiv = soup.find("div", {"id": "s-lg-guide-header-info"})
    maindiv = soup.find("div", {"id": "s-lg-guide-main"})

    if headerdiv and maindiv:
        return headerdiv.get_text() + maindiv.get_text()
    else:
        return



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
            return
    else:
        return

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

guides = glob("data/libguides/all_guides_json/*")

print("---------- ", type(guides))

# create an RDD consisting of a list of paths to .html files
guidesRDD = sc.parallelize(guides[0:10])
#guidesRDD = sc.parallelize(guides)

print("---------- guidesRDD.first():  ", guidesRDD.first())
print("---------- guidesRDD.count():  ", guidesRDD.count())

# create an RDD containing html documents
documentsRDD = guidesRDD.map(lambda guide: readguide(guide))

print("---------- documentsRDD.first():  ", documentsRDD.first())
print("---------- documentsRDD.count():  ", documentsRDD.count())

# create RDD containg plain text derived from html, guide id, and page_id
textsRDD = documentsRDD.map(lambda document: extract_text(document))

print("---------- textsRDD.count():  ", textsRDD.count())
print("---------- textsRDD.first():  ", textsRDD.first())
print("---------- textsRDD.first():  ", type(textsRDD.first()))

textsDFrame = textsRDD.toDF(["guide_id", "page_id", "words"])

textsDFrame.show(10)

texts = textsDFrame.collect()

print(" -------------- len(texts): ", len(texts))
print(" -------------- type(texts): ", type(texts))
#print(" -------------- print(texts): ", texts)

# save to parquet
textsDFrame.select("guide_id", "page_id", "words").write.save("libguides.parquet", format="parquet")

"""


"""
"""
# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(vectoredDFrame)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(res)
print("Within Set Sum of Squared Errors = " + str(wssse))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
"""
"""
texts = textsRDD.collect()

print(" -------------- len(texts): ", len(texts))
print(" -------------- type(texts): ", type(texts))
#print(" -------------- print(texts): ", texts)

joblib.dump(texts, 'data/libguides.pkl')

with open('data/libguides.json', 'w') as outfile:
    json.dump(texts, outfile)
"""
