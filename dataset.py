import numpy as np
import pandas as pd

def createDataset():
    d1 = pd.read_csv(r"dataset\NER dataset.csv", encoding = "ISO-8859-1")
    d2 = pd.read_csv(r"dataset\ner_datasetreference.csv", encoding="ISO-8859-1")
    d3 = pd.read_csv(r"dataset\ner.csv", encoding = "utf-8")
    print(d1.head(10))
    print(d2.head(10))
    print(d3.head(10))
    tags_concatenati = pd.concat([d1['Tag'], d3['labels']], ignore_index=True)
    # Estrai i valori unici
    tag_unici = tags_concatenati.unique()
    # Converti in lista, se necessario
    lista_tag_unici = list(tag_unici)
    print(lista_tag_unici)


if __name__=='__main__':
    createDataset()
