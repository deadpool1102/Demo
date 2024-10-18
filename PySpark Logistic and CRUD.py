#!/usr/bin/env python
# coding: utf-8

# PYSPARK  Using Diabetes Dataset

# In[18]:


get_ipython().system('pip install pyspark')
# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark.sql.types as tp

# Create a Spark session
spark = SparkSession.builder \
    .appName("DiabetesPredictionPipeline") \
    .getOrCreate()

####### Define the schema for the data
my_schema = tp.StructType([
    tp.StructField(name='Pregnancies', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='Glucose', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='BloodPressure', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='SkinThickness', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='Insulin', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='BMI', dataType=tp.DoubleType(), nullable=True),
    tp.StructField(name='DiabetesPedigreeFunction', dataType=tp.DoubleType(), nullable=True),
    tp.StructField(name='Age', dataType=tp.IntegerType(), nullable=True),
    tp.StructField(name='Outcome', dataType=tp.IntegerType(), nullable=True)
])

####### Read the data with the defined schema
my_data = spark.read.csv('diabetes.csv', schema=my_schema, header=True)

# Print the schema to verify structure
my_data.printSchema()

# Print the first 5 rows for quick inspection
my_data.show(5, truncate=False)

####### Replace zeros with nulls in specific columns
cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_check:
    my_data = my_data.withColumn(col, when(my_data[col] == 0, None).otherwise(my_data[col]))

# Define stages for the pipeline
imputer = Imputer(
    inputCols=cols_to_check,
    outputCols=[f"{c}_imputed" for c in cols_to_check]
).setStrategy("median")

######
assembler = VectorAssembler(
    inputCols=['Pregnancies', 'Glucose_imputed', 'BloodPressure_imputed',
               'SkinThickness_imputed', 'Insulin_imputed', 'BMI_imputed',
               'DiabetesPedigreeFunction', 'Age'],
    outputCol='features'
)

lr = LogisticRegression(featuresCol='features', labelCol='Outcome', maxIter=10)

# Create the pipeline
pipeline = Pipeline(stages=[imputer, assembler, lr])

# Split the data into training and test sets
xtrain, xtest = my_data.randomSplit([0.8, 0.2], seed=42)

# Fit the pipeline on the training data
pipeline_model = pipeline.fit(xtrain)

# Make predictions on the test data
predictions = pipeline_model.transform(xtest)

# Show predictions (first 5 rows) to verify output
predictions.select("features", "Outcome", "prediction").show(5, truncate=False)

####### Evaluate the model using AUC
evaluator = BinaryClassificationEvaluator(labelCol="Outcome", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Model AUC: {auc:.4f}")

# Calculate accuracy
correct = predictions.filter(predictions["Outcome"] == predictions["prediction"]).count()
total = predictions.count()
accuracy = correct / total
print(f"Model Accuracy: {accuracy:.4f}")

# Stop the Spark session
spark.stop()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# PYSPARK CRUD OPERATIONS

# In[20]:


get_ipython().system('pip install pyspark')

import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]") \
                    .appName('SparkByExamples.com') \
                    .getOrCreate()

data = [("James","Smith","IND","MH"),("Michael","Rose","IND","MP"), \
    ("Robert","Williams","IND","UP"),("Maria","Jones","IND","TN") \
  ]
columns=["firstname","lastname","country","state"]
df=spark.createDataFrame(data=data,schema=columns)
df.show()
print(df.collect())


states1=df.rdd.map(lambda x: x[3]).collect()
print(states1)
from collections import OrderedDict
res = list(OrderedDict.fromkeys(states1))
print(res)



#Example 2
states2=df.rdd.map(lambda x: x.state).collect()
print(states2)

states3=df.select(df.state).collect()
print(states3)

states4=df.select(df.state).rdd.flatMap(lambda x: x).collect()
print(states4)

states5=df.select(df.state).toPandas()['state']
states6=list(states5)
print(states6)

pandDF=df.select(df.state,df.firstname).toPandas()
print(list(pandDF['state']))
print(list(pandDF['firstname']))


