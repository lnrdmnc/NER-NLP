
import numpy as np
import sparknlp
from pyspark.sql import SparkSession
import pandas as pd

print("ss")
# Start Spark Session
spark = sparknlp.start()

from sparknlp.training import CoNLL


url = 'https://raw.githubusercontent.com/lnrdmnc/NER-NLP/main/dataset/ner.csv'
df = pd.read_csv(url)

# Dimensione del campione desiderato
sample_size = 20000  # Ad esempio, per ridurre il dataset a 10000 entry

# Estrai un campione casuale senza rimpiazzo
df_sample = df.sample(n=sample_size, random_state=42)

# Salva il dataset ridotto, se necessario
df_sample.to_csv('reduced_dataset.csv', index=False)
df=pd.read_csv('reduced_dataset.csv')

df.head(5)
df.info()

import re

def clean_text(text):
    text = text.lower()  # Rendi tutto minuscolo per uniformità
    text = re.sub(r"\s+", " ", text)  # Rimuovi spazi multipli
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Rimuovi caratteri speciali (opzionale)
    return text


N = 1_000
#change columns names
df.rename(columns = {'text':'sentence', 'labels':'tags'}, inplace = True)

#split train, dev , test sets
df_train, df_dev, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])

from sparknlp.training import CoNLL

training_data = CoNLL().readDataset(spark, url).limit(5000)

# Observe the first 3 rows of the Dataframe
training_data.show(3)

# Observe the first 3 rows of the Dataframe
training_data.show(3)

import pyspark.sql.functions as F

training_data.select(F.explode(F.arrays_zip(training_data.token.result,
                                            training_data.label.result)).alias("cols")) \
             .select(F.expr("cols['0']").alias("token"),
                     F.expr("cols['1']").alias("ground_truth")).groupBy('ground_truth').count().orderBy('count', ascending=False).show(100,truncate=False)

import os
os.makedirs('ner_logs', exist_ok=True)
os.makedirs('ner_graphs', exist_ok=True)

graph_folder = "./ner_graphs"

from sparknlp.annotator import TFNerDLGraphBuilder

graph_builder = TFNerDLGraphBuilder()\
              .setInputCols(["sentence", "token", "embeddings"]) \
              .setLabelColumn("label")\
              .setGraphFile("auto")\
              .setGraphFolder(graph_folder)\
              .setHiddenUnitsNumber(20)


# Import the required modules and classes
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    Tokenizer,
    SentenceDetector,
    BertEmbeddings
)

# Step 1: Transforms raw texts to `document` annotation
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

# Step 2: Getting the sentences
sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

# Step 3: Tokenization
tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

# Step 4: Bert Embeddings
embeddings = BertEmbeddings.pretrained().\
    setInputCols(["sentence", 'token']).\
    setOutputCol("embeddings")

from sparknlp.annotator import NerDLApproach

# Model training
nerTagger = NerDLApproach()\
              .setInputCols(["sentence", "token", "embeddings"])\
              .setLabelColumn("label")\
              .setOutputCol("ner")\
              .setMaxEpochs(7)\
              .setLr(0.003)\
              .setBatchSize(32)\
              .setRandomSeed(0)\
              .setVerbose(1)\
              .setValidationSplit(0.2)\
              .setEvaluationLogExtended(True) \
              .setEnableOutputLogs(True)\
              .setIncludeConfidence(True)\
              .setGraphFolder(graph_folder)\
              .setOutputLogsPath('ner_logs') 

# Define the pipeline            
ner_pipeline = Pipeline(stages=[embeddings,
                                graph_builder,
                                nerTagger])

ner_model = ner_pipeline.fit(training_data)
test_data = CoNLL().readDataset(spark, './eng.testa').limit(1000)
predictions = ner_model.transform(test_data)
predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                          predictions.label.result,
                                          predictions.ner.result)).alias("cols")) \
            .select(F.expr("cols['0']").alias("token"),
                    F.expr("cols['1']").alias("ground_truth"),
                    F.expr("cols['2']").alias("prediction")).show(30, truncate=False)
