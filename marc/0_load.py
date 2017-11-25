import pymarc
import io
from pymarc import MARCReader, JSONWriter, TextWriter
from pyspark.sql import SparkSession, Row, SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder.appName("process_marc").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

list = []
schemaString = "id marc"
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
schema = StructType(fields)

# 10,000
# marc/data/alma_bibs__2017090800_4547647970003811_new_1.xml
# 7
# marc/data/test_small_alma_bibs__2017102418_4593081760003811_new.xml

with io.open('marc/data/test_small_alma_bibs__2017102418_4593081760003811_new.xml', 'r', encoding='utf8', errors='replace') as fh:
    reader = MARCReader(fh, utf8_handling='replace', to_unicode=True, force_utf8=True, hide_utf8_warnings=True)
    for record in pymarc.parse_xml_to_array(fh):
        marc_id = record['001'].value()

        marc_string = io.StringIO()
        writer = JSONWriter(marc_string)
        writer.write(record)
        writer.close(close_fh=False)  # Important!
        list.append({"id": marc_id, "marc": marc_string.getvalue()})
        #subjects = []
        #print("---------- Begin ----------")
        for subject in record.subjects():
            #print(type(subject))
            #print(subject.value())
            #print(subject.format_field())
            #for subfield in subject:
                #print(subfield)
            subfields = subject.get_subfields('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
            #print(subfields)
            #subjects.append(subject.format_field())
        #print("---------- End ----------")
        #print(subjects)
        #return '~~'.join(subjects)
        #return subjects

df = sqlContext.createDataFrame(list, schema)

#print("df.count(): ", df.count())

#df.show(10, True)

df.write.save("marc/bibs.parquet", format="parquet")

