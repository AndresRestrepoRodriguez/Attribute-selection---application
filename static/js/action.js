$(document).ready(function() {

  $("#overview_zone").hide();
  $("#results_zone").hide();
  $('#contact').css("display", "none");

  $( "#button_load" ).click(function() {
    var form_data = new FormData($('#upload-file')[0]);
    //$('#contactForm').fadeToggle();
        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
        })
        .done(function(data) {
          //$('#contactForm').fadeToggle();
          if (data.error) {
            swal("Oops!", data.error,"error");
          }else{
            $("#load_zone").hide();
            swal("It's okey", data.success, "success");
            console.log(data.columns_name);
            console.log(data.counters);
            console.log(data.types_data);
            console.log(data.path);
            columns_names=data.columns_name
            counters=data.counters
            types_data=data.types_data
            var len_columns = columns_names.length;
            var len_types = counters.length;
            var len_counter = types_data.length;
            var pathdata = data.path;
            var criterio = data.criterio;
            for (i=0; i<len_columns; i++){
              $("#counter_data").append("<tr><td>"+columns_names[i]+"</td><td>"+counters[i]+"</td></tr>")
            }
            for (i=0; i<len_columns; i++){
              $("#type_data").append("<tr><td>"+columns_names[i]+"</td><td>"+types_data[i]+"</td></tr>")
            }
            $("#overview_zone").show();
            $.ajax({
          	  type: "GET",
          	  url: 'http://127.0.0.1:8887/datos.csv',
          	  dataType: "text",
          	  success: function(response)
          	  {
          		data = $.csv.toArrays(response);
              console.log(data);
              col = data[0]
              data.splice(0,1);
              console.log(data);
              $.extend( true, $.fn.dataTable.defaults, {
                  "searching": false,
                  "ordering": false
              } );
              $('#example').DataTable( {
                  data: data,
                  columns: col.map(function(d) { return {title:d}; })
                } );
          	  }
          	 });

             if(criterio == 'Euclidean' || criterio == 'Hamming' ){
               $("#criterio_zone").append("<p>De acuerdo a su conjuto de datos, se aplicará selección de atributos por entropía utilizando la distancia</p>")
               $("#criterio_zone").append("<p class='text-center' id='criterio_final'>"+criterio+"</p>")
             }else{
               $("#criterio_zone").append("<p>De acuerdo a su conjuto de datos, se aplicará discretización por el método</p>")
               $("#criterio_zone").append("<p class='text-center' id='criterio_final'>"+criterio+"</p>")
               $("#criterio_zone").append("<p>Para posteriormente aplicar selección de atributos por entropía utilizando la distancia Hamming</p>")
               $("#criterio_zone").append("<label class='text-center' for='labeled_option'>Seleccione el atributo referencia</label><select id='labeled_option'></select><br>")
               for(i = 0; i<len_types; i++){
                 if(types_data[i]=='nominal'){
                   $('#criterio_zone select').append("<option value=''>"+columns_names[i]+"</option>")
                 }
               }
               $("#criterio_zone").append("<br><label class='text-center' for='nivel_confianza'>Seleccione su nivel de confianza</label><input type='number' value='0.9' id='nivel_confianza' min='0.10' max='0.99' step='0.01'>")

              }
          }
        });
  });

  $( "#button_send" ).click(function() {
    var algoritmo_opcion = $('#criterio_final').text();
    var json_final = '{';
    if(algoritmo_opcion=='Chimerge'){
      var labeled_column = $( "#labeled_option option:selected" ).text();
      var confianza = $( "#nivel_confianza" ).val();
      json_final += '"option_algorithm" : '+'"'+algoritmo_opcion+'" ,';
      json_final += '"labeled_column" : '+'"'+labeled_column+'" ,';
      json_final += '"confianza" : '+'"'+confianza+'"';
    }else{
      json_final += '"option_algorithm" : '+'"'+algoritmo_opcion+'"';
    }
    json_final += '}';

    var obj_data = JSON.parse(json_final);
    $("#overview_zone").hide();
    $('#contactForm').fadeToggle();
    $.ajax({
      data : obj_data,
      type : 'POST',
      url : '/seleccionar'
    })
    .done(function(data) {
      if (data.error) {
        swal("Oops!", data.error,"error");
      }else{
        var process = data.process;
        var results = data.results;
        var suggestion = data.suggestion;
        var algoritmo = data.algoritmo;
        var len_process = process.length;
        var len_results = results.length;
        var len_suggestion = suggestion.length;
        if(algoritmo == 'Chimerge'){
          $('#process_textarea').append("Proceso discretización por Chimerge")
          $('#process_textarea').append("\n------------------------------------------------------------------------------------------\n")
          $('#process_textarea').append(process[0].join("\n"));
          $('#process_textarea').append("\n------------------------------------------------------------------------------------------\n")
          $('#process_textarea').append("Proceso selección de atributos por Hamming")
          $('#process_textarea').append("\n------------------------------------------------------------------------------------------\n")
          $('#process_textarea').append(process[1].join("\n"));
        }else{
          $('#process_textarea').append("Proceso selección de atributos por "+algoritmo)
          $('#process_textarea').append("\n------------------------------------------------------------------------------------------\n")
          $('#process_textarea').append(process.join("\n"));
        }
        //alert(results);

        //<div id='result_data' class="table-responsive">
        //  <table id="results_table" class='table table-bordered' width="100%"></table>
        //</div>

        $.extend( true, $.fn.dataTable.defaults, {
            "searching": false,
            "ordering": false,
            //"scrollY":"200px",
            //"scrollCollapse": true,
            "paging":false
        } );
        $('#results_table').DataTable( {
            data: results,
            columns: [
            { "title": "Atributos" },
            { "title": "Entopía" },
            { "title": "Diferencia" }
         ]
        });

        if(len_suggestion == 1){
          var articulo = ' el atributo ';
        }else{
          var articulo = ' los atributos ';
        }
        var atributos_suggestion = suggestion.join(' o ');
        $('#suggestion_textarea').append('Al aplicar el algoritmo de selección de atributos por entropía, se recomienda eliminar'+articulo+''+atributos_suggestion+'')
        $("#results_zone").show();
        $('#contactForm').fadeToggle();
      }
    });

  });


});

function sleep(milliseconds) {
 var start = new Date().getTime();
 for (var i = 0; i < 1e7; i++) {
  if ((new Date().getTime() - start) > milliseconds) {
   break;
  }
 }
}
