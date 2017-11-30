import pymarc
import io
import string
import os
from pymarc import MARCReader, JSONWriter, TextWriter
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, CountVectorizer, RegexTokenizer, Word2Vec

spark = SparkSession.builder.appName("process_marc").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

list = []


fields = [StructField("id", StringType(), True),
          StructField("marc", StringType(), True),
          StructField("titles", ArrayType(StringType()), True),
          StructField("subjects", ArrayType(StringType()), True),
          StructField("notes", ArrayType(StringType()), True)
          ]
schema = StructType(fields)

# 10,000
# marc/data/alma_bibs__2017090800_4547647970003811_new_1.xml
# 7
# marc/data/test_small_alma_bibs__2017102418_4593081760003811_new.xml

def array_merge(titles, subjects, notes):
    new_arr = []
    new_arr.extend(titles)
    new_arr.extend(subjects)
    new_arr.extend(notes)
    return new_arr

exclude = set(string.punctuation)

def xstr(s):
    global exclude
    if s is None:
        return ''
    return ''.join(ch for ch in s if ch not in exclude)

# Define udf
from pyspark.sql.functions import udf
udf_array_merge = udf(array_merge, ArrayType(StringType()))

path = 'marc/data'
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    #tree = ET.parse(fullname)

    #with io.open('marc/data/alma_bibs__2017090800_4547647970003811_new_1.xml', 'r', encoding='utf8', errors='replace') as fh:
    with io.open(fullname, 'r', encoding='utf8',
                 errors='replace') as fh:
        reader = MARCReader(fh, utf8_handling='replace', to_unicode=True, force_utf8=True, hide_utf8_warnings=True)
        for record in pymarc.parse_xml_to_array(fh):

            marc_id = record['001'].value()

            marc_string = io.StringIO()
            writer = JSONWriter(marc_string)
            writer.write(record)
            writer.close(close_fh=False)  # Important!

            titles = [record.title()]

            subjects = []
            notes = []

            for subject in record.subjects():
                #print(type(subject))
                #print(subject.value())
                #print(subject.format_field())
                #for subfield in subject:
                    #print(subfield)
                subfields = subject.get_subfields('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
                subjects.extend(subfields)


            # extract_marc("500a:508a:511a:515a:518a:521ab:530abcd:533abcdefmn:534pabcefklmnt:538aiu:546ab:550a:586a:588a")
            if record['500']:
                notes.extend([record['500']['a']])

            if record['502']:
                notes.extend([record['502']['a'], record['502']['b'], record['502']['c'], record['502']['g'],
                         record['502']['o']])

            if record['505']:
                notes.extend([record['505']['a'], record['505']['g'], record['505']['r'], record['505']['t']])

            if record['508']:
                notes.extend([record['508']['a']])

            if record['511']:
                notes.extend([record['511']['a']])

            if record['515']:
                notes.extend([record['515']['a']])

            if record['518']:
                notes.extend([record['518']['a']])

            #extract_marc("500a:508a:511a:515a:518a:521ab:530abcd:533abcdefmn:534pabcefklmnt:538aiu:546ab:550a:586a:588a")


            list.append({"id": marc_id,
                         "marc": marc_string.getvalue(),
                         "titles": [xstr(x).lower() if len(titles) else '' for x in titles],
                         "subjects": [xstr(x).lower() if len(subjects) else '' for x in subjects],
                         "notes": [xstr(x).lower() if len(notes) else '' for x in notes]})

                #print(subfields)
                #subjects.append(subject.format_field())
            #print("---------- End ----------")
            #print(subjects)
            #return '~~'.join(subjects)
            #return subjects

df = sqlContext.createDataFrame(list, schema)
df.drop("marc").show(10, True)
print(df.printSchema())

#print("df.count(): ", df.count())

new = df.select("id", "marc", "titles", "subjects", "notes").withColumn("allTextArray", udf_array_merge(df.titles, df.subjects, df.notes))
new.show(10, True)
print(new.printSchema())

new.write.save("marc/bibs.parquet", format="parquet")

print("new.count(): ", new.count())