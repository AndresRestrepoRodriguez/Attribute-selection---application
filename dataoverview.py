import pandas as pd
import numpy as np
from scipy.io import arff as ar
import arff

def leer_arff(ruta):
    _, meta = ar.loadarff(ruta)
    columns = meta.names()
    typescol = meta.types()
    datad = arff.load(open(ruta, 'r'))
    dataframe = pd.DataFrame(datad['data'], columns=columns)
    dataframe.to_csv('D:/Maestria/SemestreII/MineriaDatos/appselection/static/data/datos.csv',index=False)
    counter_data = dataframe.count().values.tolist()
    criterio = generar_criterio(typescol)
    return columns, counter_data, typescol, criterio


def generar_criterio(tipo_atributos):
    set_atributos = set(tipo_atributos)
    result = len(set_atributos) == 1
    tipo_set = set_atributos.pop()
    if(result and tipo_set=='numeric'):
        criterio = 'Euclidean'
    elif(result and tipo_set=='nominal'):
        criterio = 'Hamming'
    else:
        criterio = 'Chimerge'
    return criterio

def leer_csv(ruta):
    dataframe = pd.read_csv(ruta)
    return dataframe

def get_types(dataframe):
    types_list=[]
    for i in dataframe.columns:
        if(dataframe[i].dtype == np.float64 or dataframe[i].dtype == np.int64):
            types_list.append('numeric')
        else:
            types_list.append('categoric')
    return types_list

def get_initial_values(dataframe):
    columns_name = dataframe.columns.values.tolist()
    counter_data = dataframe.count().values.tolist()
    types_data = get_types(dataframe)
    return columns_name, counter_data, types_data
