# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import chi2
import math

def chimerge_discretization_individual(dataframe, numeric_column, labeled_attribute, confianza_parametro):
    datasort = dataframe[[numeric_column, labeled_attribute]]
    datasort.sort_values(by=[numeric_column], inplace=True)
    num_c = numeric_column
    to_disc = datasort[numeric_column].values
    labeled = datasort[labeled_attribute].values
    #----- intervalos
    min_to_disc = np.amin(to_disc)
    max_to_disc = np.amax(to_disc)
    diff_mm = max_to_disc-min_to_disc
    num_particion=6
    val_int = diff_mm/num_particion
    array_intervalos_iniciales = []
    aumento = 0.01
    inicial = min_to_disc
    for i in range(num_particion):
        array_temp = []
        array_temp.append(inicial)
        siguiente = round((inicial + val_int),2)
        array_temp.append(siguiente)
        inicial = round((siguiente+aumento),2)
        array_intervalos_iniciales.append(array_temp)
    #-------- calcular umbral
    clases_labeled = list(set(labeled))
    num_clases = len(clases_labeled)
    grados_libertad = num_clases-1
    confianza = confianza_parametro
    umbral = chi2.ppf(confianza, grados_libertad)
    #----------- matriz de frecuencias
    intervalos_finales = []
    print(array_intervalos_iniciales)
    while(True):
        mat_frecuencia = np.empty([2,num_clases])
        for i in range(2):
            print(array_intervalos_iniciales[i])
            for j in range(num_clases):
                df_aux = datasort.loc[(datasort[numeric_column] >= array_intervalos_iniciales[i][0]) & (datasort[numeric_column] <= array_intervalos_iniciales[i][1])]
                df_aux_labeled = df_aux.loc[(df_aux[labeled_attribute]==clases_labeled[j])]
                mat_frecuencia[i][j]=len(df_aux_labeled)
            R_suma = np.sum(mat_frecuencia, axis=1)
            C_suma = np.sum(mat_frecuencia, axis=0)
            N = np.sum(C_suma)

            array_X = []
            for k in range(mat_frecuencia.shape[0]):
                for m in range(mat_frecuencia.shape[1]):
                    E_temp = (R_suma[k]*C_suma[m])/N
                    if(E_temp == 0):
                        E_temp = 0.1
                    valor_temp = (math.pow(((mat_frecuencia[k][m])-E_temp),2))/E_temp
                    array_X.append(valor_temp)
                X_final = sum(array_X)
        print(X_final)
        if(X_final <= umbral):
            print('unir')
            int_unificado = [array_intervalos_iniciales[i-1][0],array_intervalos_iniciales[i][1]]
            array_intervalor_iniciales_temp = []
            array_intervalor_iniciales_temp.append(int_unificado)
            array_intervalor_iniciales_temp.extend(array_intervalos_iniciales[i+1:])
            array_intervalos_iniciales = array_intervalor_iniciales_temp.copy()
            print(array_intervalos_iniciales)
            if(len(array_intervalos_iniciales)==1):
                intervalos_finales.append(array_intervalos_iniciales[0])
                return intervalos_finales
        else:
            intervalos_finales.append(array_intervalos_iniciales[i-1])
            del array_intervalos_iniciales[i-1]
            if(len(array_intervalos_iniciales) == 1):
                intervalos_finales.append(array_intervalos_iniciales[0])
                return intervalos_finales
            if(len(array_intervalos_iniciales) == 0):
                return intervalos_finales

def replace_discretization(intervalos, numeric_column, dataframe):
    data_col = dataframe[numeric_column].values
    data_col_nuevo = np.zeros_like(data_col).astype(np.str)
    for i in range(len(data_col)):
        for k in range(len(intervalos)):
            if(data_col[i]>=intervalos[k][0] and data_col[i]<=intervalos[k][1]):
                data_col_nuevo[i] = str(intervalos[k])
    dataframe[numeric_column] = data_col_nuevo

def get_numeric_columns(name_cols, type_cols):
    array_numeric_cols = []
    for i in range(len(type_cols)):
        if(type_cols[i] == 'numeric'):
            array_numeric_cols.append(name_cols[i])
    return array_numeric_cols

def chimerge_general(dataframe, numeric_columns, labeled_attribute, confianza_p):
    print(numeric_columns)
    array_process = []
    for k in range(len(numeric_columns)):
        array_process.append('Discretización del atributo: '+numeric_columns[k])
        print(numeric_columns[k])
        intervalos = chimerge_discretization_individual(dataframe, numeric_columns[k], labeled_attribute, confianza_p)
        array_process.append('Intervalos obtenidos: '+str(intervalos))
        print(intervalos)
        replace_discretization(intervalos, numeric_columns[k], dataframe)
    return dataframe, array_process
