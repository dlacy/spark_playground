from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, CountVectorizer, RegexTokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("tokenizer").getOrCreate()

# Loads data.
raw = spark.read.load("data/libguides_bow.parquet")
raw.show(10)
#print(" ----------- raw.count()", raw.count())

guidesDFrame = raw.filter("words is not null")
#print(" ----------- guidesDFrame.count()", guidesDFrame.count())

# begin tokenization and other processing

tokenizer = Tokenizer(inputCol="words", outputCol="bow")
TokenizerData = tokenizer.transform(guidesDFrame)
TokenizerData.show(10)
guidesDFrame = TokenizerData


# Remove Stop Words
# https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover
remover = StopWordsRemover(inputCol="bow", outputCol="stop_removed")
StopWordsRemoverData = remover.transform(guidesDFrame)
StopWordsRemoverData.show(10)
guidesDFrame = StopWordsRemoverData

#https://spark.apache.org/docs/latest/ml-features.html#tf-idf
hashingTF = HashingTF(inputCol="stop_removed", outputCol="HashingTF", numFeatures=1000)
featurizedData = hashingTF.transform(StopWordsRemoverData)
guidesDFrame = featurizedData

#
cv = CountVectorizer(inputCol="stop_removed", outputCol="CountVectorizer", vocabSize=1000, minDF=1.0, minTF=2.0)
transformer = cv.fit(guidesDFrame)
print(" ----------- ", transformer.vocabulary)
CountVectorizerData = transformer.transform(guidesDFrame)
guidesDFrame = CountVectorizerData

idf = IDF(inputCol="HashingTF", outputCol="IDF_HashingTF")
idfModel = idf.fit(guidesDFrame)
rescaledData = idfModel.transform(guidesDFrame)
guidesDFrame = rescaledData

idf = IDF(inputCol="CountVectorizer", outputCol="IDF_CountVectorizer")
idfModel = idf.fit(guidesDFrame)
rescaledData = idfModel.transform(guidesDFrame)
guidesDFrame = rescaledData

guidesDFrame.write.save("data/guides_tokenized_and_vectorized.parquet", format="parquet")
#guidesDFrame.select("guide_id", "page_id", "HashingTF", "CountVectorizer", "IDF_HashingTF", "IDF_CountVectorizer").show(truncate=False)

