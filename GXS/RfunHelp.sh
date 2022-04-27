#!/bin/bash

#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
funname=$1
if [ -z "$funname" ]; then
	echo "ERROR: No se ha indicado ningun nombre de funcion."
	exit 101
fi
#$$$$$$$$$$$$$
#$ Variables $
#$$$$$$$$$$$$$
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
found=false
anchor=" = function("
funcform="91m"
textform="37m"
normform="0m"
#$$$$$$$$$$$$$
#$ Ejecucion $
#$$$$$$$$$$$$$
# Procesar ficheros .R en el directorio actual:
grepcom="$funname.*$anchor"
for file in $DIR/*.R; do
	if grep -q "$grepcom" $file; then
		# Registrar limites de funcion:
		inifun=$(grep -n "$grepcom" $file | cut -d : -f 1)
		initemp=$(grep -n "$anchor" $file | cut -d : -f 1 |  tr "\n" " ")
		for line in ${initemp[@]}; do
			if [ $line -gt $inifun ]; then
				ininext=$line
				break
			fi
		done
		if [ -z "$ininext" ]; then ininext=$(($(cat $file | wc -l)+1));fi
		finfun=$(grep -Fn "}" $file | cut -d : -f 1 |  tr "\n" " ")
		for line in ${finfun[@]}; do
			if [ $line -gt $inifun ] && [ $line -lt $ininext ]; then
				finfun=$(($line))
				break
			fi
		done
		# Registrar limites de texto de ayuda:
		inihelp=$(($inifun+1))
		finhelp=$(grep -Fn "  #%%%%%" $file | cut -d : -f 1 | tr "\n" " ")
		for line in ${finhelp[@]}; do
			if [ $line -lt $finfun ]; then
				finhelp=$line
			fi
		done
		# Capturar funcion y texto de ayuda:
		sedcom=$inifun"p"
		functext=$(sed -n $sedcom $file)
		sedcom=$inihelp","$finhelp"p"
		helptext=$(sed -n $sedcom $file)
		# Imprimir texto de ayuda:
		echo -e "\e[$funcform$functext\e[$normform"
		echo -e "\e[$textform$helptext\e[$normform"
		# Terminar bucle:
		found=$true
		break
	fi
done
# Terminar si no se encuentran coincidencias:
if [ "$found" = "false" ]; then
	echo "ERROR: No se ha encontrado la funcion '$funname' en ningun archivo .R de '$DIR'."
fi
