from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TfIdfExample").getOrCreate()

# Loads data.
df = spark.read.load("data/libguides_bow.parquet")
df.show(10)

# begin tokenization and other processing

# Remove Stop Words
# https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover
transformer = StopWordsRemover(inputCol="words", outputCol="stop_removed")
StopWordsRemoverData = transformer.transform(df)
StopWordsRemoverData.show(10)
df = StopWordsRemoverData

# TF-IDF
# https://spark.apache.org/docs/latest/ml-features.html#tf-idf
transformer = HashingTF(inputCol="stop_removed", outputCol="TF", numFeatures=20)
HashingTFData = transformer.transform(df)
HashingTFData.show(10)
df = HashingTFData


# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="stop_removed", outputCol="cv_features", vocabSize=10, minDF=2.0, minTF=3.0)
transformer = cv.fit(df)
CountVectorizerData = transformer.transform(df)

CountVectorizerData.show(10)
df = CountVectorizerData

