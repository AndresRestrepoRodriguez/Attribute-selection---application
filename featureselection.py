from scipy.io import arff
import pandas as pd
import numpy as np
import math

def calculate_matrix_euclidean(data):
    alpha = 0.5
    maxFeatures = data.max()
    minFeatures = data.min()
    diffFeatures = (maxFeatures-minFeatures).values
    indexDf = data.index.values
    dfTestValues = data.values
    matrixRes = np.zeros((len(indexDf),len(indexDf)))
    for i in range(len(indexDf)):
        for j in range(i+1,(len(indexDf))):
            array_temp = []
            for k in range(len(dfTestValues[i])):
                valor_temp = pow((dfTestValues[i][k]-dfTestValues[j][k])/diffFeatures[k],2)
                array_temp.append(valor_temp)
            valor_Dij = sum(array_temp)
            valor_Dij = math.sqrt(valor_Dij)
            valor_matij = pow(math.e,((-1)*alpha*(valor_Dij)))
            matrixRes[i][j] = valor_matij
    return matrixRes

def calculate_matrix_hamming(data):
    n = len(data.columns)
    indexDf = data.index.values
    dfTestValues = data.values
    matrixRes = np.zeros((len(indexDf),len(indexDf)))
    for i in range(len(indexDf)):
        for j in range(i+1,(len(indexDf))):
            array_temp = []
            for k in range(len(dfTestValues[i])):
                if(dfTestValues[i][k] == dfTestValues[j][k]):
                    valor_temp=1;
                else:
                    valor_temp=0;
                array_temp.append(valor_temp)
            valor_matij = sum(array_temp)/n
            matrixRes[i][j] = valor_matij
    return matrixRes

def calculate_entropy(matrix):
    matrix_local = matrix.copy()
    for i in range(len(matrix_local)):
        for j in range(len(matrix_local)):
            if(matrix_local[i][j] != 0 and matrix_local[i][j] != 1):
                valor_temp = (matrix_local[i][j]*math.log2(matrix_local[i][j]))+((1-matrix_local[i][j])*math.log2((1-matrix_local[i][j])))
                matrix_local[i][j] = (-1)*valor_temp
            elif(matrix_local[i][j] == 1):
                matrix_local[i][j] = 0
    array_temp_entropy = []
    for i in range(len(matrix_local)):
        valor_temp = sum(matrix_local[i])
        array_temp_entropy.append(valor_temp)
    entropy = sum(array_temp_entropy)
    return entropy

def suggestion_generation(resultados_diff, resultado_entropias):
    array_atributos = []
    min_diff = min(resultados_diff)
    for i in range(len(resultado_entropias)):
        if(resultado_entropias[i][2] == min_diff):
            array_atributos.append(resultado_entropias[i][0][4:])
    return array_atributos

def feature_selection_euclidean(data):
    array_process = []
    array_temp = []
    array_results = []
    array_diff = []
    columns = data.columns.values
    resultado_general = calculate_matrix_euclidean(data)
    array_process.append('Cálculo de matriz de distancia Euclideana con la totalidad de datos')
    entropia_general = calculate_entropy(resultado_general)
    array_process.append('Cálculo de entropía con la totalidad de datos')
    array_temp.append('General')
    array_temp.append(round(entropia_general,2))
    array_temp.append(round((entropia_general-entropia_general),2))
    array_results.append(array_temp)
    for i in range(len(columns)):
        array_temp = []
        df_aux = data.drop([columns[i]], axis=1)
        resultado_particular = calculate_matrix_euclidean(df_aux)
        array_process.append(f'Cálculo de matriz de distancia Euclideana sin el atributo {columns[i]}')
        entropia_particular = calculate_entropy(resultado_particular)
        entropia_diff = abs(entropia_general - entropia_particular)
        array_process.append(f'Cálculo de entropía sin el atributo {columns[i]}')
        array_temp.append(f'Sin {columns[i]}')
        array_temp.append(round(entropia_particular,2))
        array_temp.append(round(entropia_diff,2))
        array_diff.append(round(entropia_diff,2))
        array_results.append(array_temp)
    array_suggestion = suggestion_generation(array_diff,array_results)
    return array_process, array_results, array_suggestion

def feature_selection_hamming(data):
    array_process = []
    array_temp = []
    array_results = []
    array_diff = []
    columns = data.columns.values
    resultado_general = calculate_matrix_hamming(data)
    array_process.append('Cálculo de matriz de distancia Hamming con la totalidad de datos')
    entropia_general = calculate_entropy(resultado_general)
    array_process.append('Cálculo de entropía con la totalidad de datos')
    array_temp.append('General')
    array_temp.append(round(entropia_general,2))
    array_temp.append(round((entropia_general-entropia_general),2))
    array_results.append(array_temp)
    for i in range(len(columns)):
        array_temp = []
        df_aux = data.drop([columns[i]], axis=1)
        resultado_particular = calculate_matrix_hamming(df_aux)
        array_process.append(f'Cálculo de matriz de distancia Hamming sin el atributo {columns[i]}')
        entropia_particular = calculate_entropy(resultado_particular)
        array_process.append(f'Cálculo de entropía sin el atributo {columns[i]}')
        entropia_diff = abs(entropia_general - entropia_particular)
        array_temp.append(f'Sin {columns[i]}')
        array_temp.append(round(entropia_particular,2))
        array_temp.append(round(entropia_diff,2))
        array_diff.append(round(entropia_diff,2))
        array_results.append(array_temp)
    array_suggestion = suggestion_generation(array_diff,array_results)
    return array_process, array_results, array_suggestion
