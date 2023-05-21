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
patron="La media calculada en función de los indices es: ([0-9.]+)"
patron2="La varianza resultante es: ([0-9.]+)"
patron3="La desviacion tipica resultante es: ([0-9.]+)"
echo " ;Media; Varianza; Desviacion" > "$salida"

# Variables para rastrear las coincidencias
nlinea=0
ultima_coincidencia=0
actual=0
itern=0
media=0
varianza=0
desviacion=0 
# Obtener el número total de líneas en el archivo de texto
total_lineas=$(wc -l < "$archivo")
nlinea=0
# Leer el archivo de texto línea por línea
while IFS= read -r linea; do
  ((nlinea++))
  ((progreso_global++))
  
  # Comprobar si la línea coincide con el patrón
  if [[ $linea =~ $patron ]]; then
    media="${BASH_REMATCH[1]}"
  fi
  if [[ $linea =~ $patron2 ]]; then
    varianza="${BASH_REMATCH[1]}"
  fi
  if [[ $linea =~ $patron3 ]]; then
    desviacion="${BASH_REMATCH[1]}"
    ((itern++))
    # Escribir los valores en el archivo de Excel
    echo "iter: $itern" >> "$salida"
    echo " ;$media; $varianza; $desviacion" >> "$salida"
  fi

  # Imprimir la barra de progreso actualizada del archivo
  bar_length=50
  filled=$(( bar_length * nlinea / total_lineas ))
  unfilled=$(( bar_length - filled ))
  percent=$(( nlinea * 100/ total_lineas ))
  printf "\r[Progreso de archivo: [%s%s] %d%%" "$(printf '█%.0s' $(seq 1 $filled))" "$(printf ' %.0s' $(seq 1 $unfilled))" "$percent"
done < "$archivo"

echo -e "\n[Fichero procesado con éxito] ¡Datos guardados en $salida!"

