from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, CountVectorizer, RegexTokenizer, Word2Vec
from pyspark.sql import SparkSession
from pyspark.ml.clustering import LDA

spark = SparkSession.builder.appName("tokenizer").getOrCreate()

# Loads data.
raw = spark.read.load("data/libguides_txt.parquet")

# Purge Null / None datas
nullified = raw.na.drop()

guidesDFrame = nullified.select("guide_id","guide_name","page_id","page_name","words")

tokenizer = Tokenizer(inputCol="words", outputCol="word_tokens")
TokenizerData = tokenizer.transform(guidesDFrame)
guidesDFrame = TokenizerData


# Remove Stop Words
# https://spark.apache.org/docs/latest/ml-features.html#stopwordsremover
remover = StopWordsRemover(inputCol="word_tokens", outputCol="stop_removed")

my_sw = ["guide", "books", "database", "meta", "results", "https", "login", "updated", "david", "dillard", "use", "guide", "www", "search", "edu", "guides", "eric", "library", "find", "check", "doc", "check", "administration", "want", "ebsco", "http", "r", "f", "google", "com", "less", "tinyurl", "isbn", "call", "number", "date", "c", "paley", "temple", "research"]
sw = remover.loadDefaultStopWords("english")
remover.setStopWords(sw + my_sw)

StopWordsRemoverData = remover.transform(guidesDFrame)
guidesDFrame = StopWordsRemoverData

cv = CountVectorizer(inputCol="stop_removed", outputCol="CountVectorizer", vocabSize=1000, minDF=1.0, minTF=10.0)
transformer = cv.fit(guidesDFrame)
print(" ----------- ", transformer.vocabulary)

vacabulary = transformer.vocabulary

CountVectorizerData = transformer.transform(guidesDFrame)
guidesDFrame = CountVectorizerData

# Trains a LDA model.
lda = LDA(k=10, maxIter=15, featuresCol="CountVectorizer")
model = lda.fit(guidesDFrame)

print("------------")
model.vocabSize()
print("------------")
model.describeTopics(maxTermsPerTopic=20).show()
topics = model.describeTopics(maxTermsPerTopic=20).collect()
print(topics)

i=0
for topic in topics:
    print(topic["topic"])
    for word_id in topic["termIndices"]:
        print(word_id, " - ", vacabulary[word_id])
print("------------")

ldaData = model.transform(guidesDFrame)
guidesDFrame = ldaData

guidesDFrame.show(10)
