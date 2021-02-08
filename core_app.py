from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
import dataoverview as do
import featureselection as fs
import chimerge as ch


UPLOAD_FOLDER = '/appselection/static/data/'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['arff'])

ruta_datos = ''
ruta_datos_csv = '/static/data/datos.csv'
columns_name = []
types_data = []
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
	return render_template('principal.html')

@app.route('/uploadajax', methods=['POST'])
def model():
    global ruta_datos
    global columns_names
    global types_data
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'datos.arff'))
    ruta_datos = UPLOAD_FOLDER+'datos.arff'
    columns_names, counters, types_data, criterio = do.leer_arff(ruta_datos)

    if (True) :
        return jsonify({ 'success' :  'Data loaded successfully', 'columns_name':columns_names, 'counters':counters, 'types_data':types_data, 'path':ruta_datos, 'criterio':criterio})
    else:
        return jsonify({'error' : 'Data error'})

@app.route('/seleccionar', methods=['POST'])
def seleccion():
    global ruta_datos_csv
    global columns_names
    global types_data
    data = request.form
    data_dictionary = data.copy()
    algoritmo_opcion = data_dictionary['option_algorithm']
    if(algoritmo_opcion == 'Chimerge'):
        col_labeled = data_dictionary['labeled_column']
        confianza = float(data_dictionary['confianza'])
        dataframe = do.leer_csv(ruta_datos_csv)
        numeric_columns = ch.get_numeric_columns(columns_names,types_data)
        discreted_dataframe, process_chimerge = ch.chimerge_general(dataframe, numeric_columns, col_labeled,confianza)
        #print(dataframe)
        process, results, suggestion = fs.feature_selection_hamming(discreted_dataframe)
        process = [process_chimerge, process]

    elif(algoritmo_opcion == 'Euclidean'):
        dataframe = do.leer_csv(ruta_datos_csv)
        process, results, suggestion = fs.feature_selection_euclidean(dataframe)
    else:
        dataframe = do.leer_csv(ruta_datos_csv)
        process, results, suggestion = fs.feature_selection_hamming(dataframe)

    if (True) :
        return jsonify({ 'success' :  'successfully', 'process': process, 'results':results, 'suggestion':suggestion, 'algoritmo':algoritmo_opcion})
    else:
        return jsonify({'error' : 'error'})

if __name__ == '__main__':
	app.run(debug=False)
