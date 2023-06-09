#!/bin/bash

if [ $# -eq 0 ]; then
  echo "[ERROR]: Falta directorio como parámetro; ejemplo de ejecución: ./proc_data.sh directorio/"
  exit 1
else
  # Obtener el nombre del directorio del primer parámetro
  directorio="$1"
  if [ ! -d "$directorio" ]; then
    echo "[ERROR]: El directorio no existe"
    exit 1
  fi
fi



# Expresión regular para extraer los datos de interés
patron="Neuron ([0-9]+) likes note ([0-9]+), ([0-9.]+)% success.*incorrect correct total.*([0-9]+) , ([0-9]+) , ([0-9]+)"


# Variables para rastrear las coincidencias
nlinea=0
ultima_coincidencia=0
actual=0
itern=0
num_archivos=0
progreso_global=0

# Obtener el número total de archivos en el directorio
num_archivos_totales=$(find "$directorio" -maxdepth 1 -type f -name "*.txt" | wc -l)

echo "Procesando archivos en el directorio $directorio ..."

# Leer cada archivo .txt en el directorio
for archivo in "$directorio"/*.txt; do
  ((num_archivos++))
  itern=0
  echo "Procesando archivo: $(basename "$archivo") [$num_archivos/$num_archivos_totales]"
  nombre_archivo=$(basename "$archivo" .txt)
  if [ ! -d "$directorio""out_xlsx" ]; then
    mkdir "$directorio""out_xlsx"
  fi
  salida="$directorio""out_xlsx/${nombre_archivo}.xlsx"
  echo " ;Neuron N; Note; % success; incorrect spikes; correct spikes; total spikes" > "$salida"
  
  # Obtener el número total de líneas en el archivo de texto
  total_lineas=$(wc -l < "$archivo")
  nlinea=0

  # Leer el archivo de texto línea por línea
  while IFS= read -r linea; do
    ((nlinea++))
    ((progreso_global++))
    
    # Comprobar si la línea coincide con el patrón
    if [[ $linea =~ $patron ]]; then
      # Extraer los valores correspondientes
      neuron="${BASH_REMATCH[1]}"
      note="${BASH_REMATCH[2]}"
      success="${BASH_REMATCH[3]}"
      incorrect="${BASH_REMATCH[4]}"
      correct="${BASH_REMATCH[5]}"
      total="${BASH_REMATCH[6]}"
      
      # Escribir los valores en el archivo de Excel
      echo " ;$neuron; $note; $success; $incorrect; $correct; $total" >> "$salida"
      
      # Actualizar la última coincidencia encontrada
      ultima_coincidencia=$nlinea
    else
      # Comprobar si han pasado dos líneas vacías desde la última coincidencia
      actual=$nlinea
      if ((actual - ultima_coincidencia == 2 && actual > 6)); then
        ((itern++))
        echo "Iter number: $itern" >> "$salida"
      fi
    fi

    # Imprimir la barra de progreso actualizada del archivo
    bar_length=50
    filled=$(( bar_length * nlinea / total_lineas ))
    unfilled=$(( bar_length - filled ))
    percent=$(( nlinea * 100/ total_lineas ))
    printf "\r[Progreso de archivo: [%s%s] %d%%" "$(printf '█%.0s' $(seq 1 $filled))" "$(printf ' %.0s' $(seq 1 $unfilled))" "$percent"
  done < "$archivo"

echo -e "\n[Fichero procesado con éxito] ¡Datos guardados en $salida!"
done

total_lineas_globales=$(find "$directorio" -maxdepth 1 -type f -name "*.txt" -exec cat {} + | wc -l)
bar_length_global=50
filled_global=$(( bar_length_global * progreso_global / total_lineas_globales ))
unfilled_global=$(( bar_length_global - filled_global ))
percent_global=$(( progreso_global * 100 / total_lineas_globales ))
printf "\n[Progreso global: [%s%s] %d%%\n" "$(printf '█%.0s' $(seq 1 $filled_global))" "$(printf ' %.0s' $(seq 1 $unfilled_global))" "$percent_global"

echo "[Proceso completado] Todos los archivos procesados y los datos guardados en $directorio""out_xlsx"
