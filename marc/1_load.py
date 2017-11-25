import pymarc
import io
import json
from pymarc import MARCReader, JSONWriter, TextWriter, JSONReader
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, CountVectorizer, RegexTokenizer, Word2Vec
from pyspark.ml.clustering import LDA

spark = SparkSession.builder.appName("process_marc").getOrCreate()

df = spark.read.load("marc/bibs.parquet")

#df.show(10)


def get_title(marc):
    marc_io = io.StringIO()
    marc_io.write(marc)

    reader = JSONReader(marc)
    for record in reader:
        return record.title()

def get_subjects(marc):
    marc_io = io.StringIO()
    marc_io.write(marc)
    reader = JSONReader(marc)
    subjects = []
    for record in reader:
        for subject in record.subjects():
            subfields = subject.get_subfields('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                                              'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
            subjects.extend(subfields)
        return subjects

def gt(topicDistribution):
    global vocabulary
    global topics
    #global index
    c = [b[0] for b in sorted(enumerate(topicDistribution), key=lambda i:i[1])]
    index = 3
    words = []
    for word_id in topics[c[index]]["termIndices"]:
        words.append(vacabulary[word_id])
    dict = {"cluster_id": c[index], "score": topicDistribution[c[index]], "value": " ".join(words)}
    dict_str = json.dumps(dict)
    return json.loads(dict_str)

def get_topic(index, topicDistribution):
    global vocabulary
    global topics
    #global index
    c = [b[0] for b in sorted(enumerate(topicDistribution), key=lambda i:i[1])]
    #index = 3
    words = []
    for word_id in topics[c[index]]["termIndices"]:
        words.append(vacabulary[word_id])
    dict = {"cluster_id": c[index], "score": topicDistribution[c[index]], "value": " ".join(words)}
    dict_str = json.dumps(dict)
    return json.loads(dict_str)

def get_topic_3(topicDistribution):
    return get_topic(3, topicDistribution)

def get_topic_2(topicDistribution):
    return get_topic(2, topicDistribution)

def get_topic_1(topicDistribution):
    return get_topic(1, topicDistribution)

def get_topic_0(topicDistribution):
    return get_topic(0, topicDistribution)

def flatten_subjects(subjects):
    return " ".join(subjects)

# Define udf
from pyspark.sql.functions import udf
udf_get_title = udf(get_title, StringType())
udf_get_subjects = udf(get_subjects, ArrayType(StringType()))
udf_flatten_subjects = udf(flatten_subjects, StringType())
udf_get_topic = udf(gt)
udf_get_topic_3 = udf(get_topic_3, StructType([StructField("cluster_id", IntegerType()), StructField("score", FloatType()), StructField("value", StringType())]))
udf_get_topic_2 = udf(get_topic_2, StructType([StructField("cluster_id", IntegerType()), StructField("score", FloatType()), StructField("value", StringType())]))
udf_get_topic_1 = udf(get_topic_1, StructType([StructField("cluster_id", IntegerType()), StructField("score", FloatType()), StructField("value", StringType())]))
udf_get_topic_0 = udf(get_topic_0, StructType([StructField("cluster_id", IntegerType()), StructField("score", FloatType()), StructField("value", StringType())]))

newDF = df.withColumn("title", udf_get_title(df.marc)).withColumn("marc_subjects", udf_get_subjects(df.marc))
df = newDF.withColumn("full_text", udf_flatten_subjects(newDF.marc_subjects))

tokenizer = RegexTokenizer(inputCol="full_text", outputCol="word_tokens", pattern="\\W")
TokenizerData = tokenizer.transform(df)
df = TokenizerData

remover = StopWordsRemover(inputCol="word_tokens", outputCol="stop_removed")
my_sw = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
sw = remover.loadDefaultStopWords("english")
remover.setStopWords(sw + my_sw)
StopWordsRemoverData = remover.transform(df)
df = StopWordsRemoverData

cv = CountVectorizer(inputCol="stop_removed", outputCol="CountVectorizer", vocabSize=1000, minDF=1.0, minTF=3.0)
transformer = cv.fit(df)
print(" ----------- ", transformer.vocabulary)
vacabulary = transformer.vocabulary
CountVectorizerData = transformer.transform(df)
df = CountVectorizerData

# Trains a LDA model.
lda = LDA(k=4, maxIter=15, featuresCol="CountVectorizer")
model = lda.fit(df)
print("------------")
model.vocabSize()
print("------------")
model.describeTopics(maxTermsPerTopic=10).show(10, False)
topics = model.describeTopics(maxTermsPerTopic=10).collect()
print(topics)

for topic in topics:
    print(topic["topic"])
    for word_id in topic["termIndices"]:
        print(word_id, " - ", vacabulary[word_id])
print("------------")

ldaData = model.transform(df)
df = ldaData

#newDF = df.drop("marc")
df.show(10, True)
print("df.count(): ", df.count())
print(df.printSchema())

index = 0

"""
test = df.select("id", "marc_subjects", "topicDistribution")\
    .withColumn("topic_3", udf_get_topic(df.topicDistribution))
"""

test = df.select("id", "marc_subjects", "topicDistribution")\
    .withColumn("topic_3", udf_get_topic_3(df.topicDistribution))\
    .withColumn("topic_2", udf_get_topic_2(df.topicDistribution))\
    .withColumn("topic_1", udf_get_topic_1(df.topicDistribution))\
    .withColumn("topic_0", udf_get_topic_0(df.topicDistribution))

test.show(10, False)

df = test.drop("topicDistribution")
print(df.printSchema())


records = df.write\
    .format("org.elasticsearch.spark.sql")\
    .option("es.nodes", "localhost")\
    .option("es.port", "9200")\
    .option("es.nodes.wan.only", "true")\
    .option("es.resouce.auto.create", "marc/test")\
    .option("es.read.field.as.array.include", "marc_subjects")\
    .mode("append")\
    .save("marc/record")


#    .option("es.input.json", "true")\
#
#   .option("es.index.auto.create", "true")\
#    .option("es.mapping.id", "id")\


