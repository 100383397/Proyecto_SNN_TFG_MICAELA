#!/bin/bash

if [ $# -eq 0 ]; then
  echo "[ERROR]: Falta fichero como parámetro; ejemplo de ejecución: ./proc_data.sh out.txt"
  exit 1
else
  archivo="$1"
  if [ ! -f "$archivo" ]; then
    echo "El archivo $archivo no existe."
    exit 1
  fi
fi

salida="medidas.xlsx"
# Expresión regular para extraer los datos de interés
patron="Neuron ([0-9]+) likes note ([0-9]+), ([0-9.]+)% success.*incorrect correct total.*([0-9]+) , ([0-9]+) , ([0-9]+)"


# Variables para rastrear las coincidencias
nlinea=0
ultima_coincidencia=0
actual=0
itern=0
num_archivos=0
progreso_global=0


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