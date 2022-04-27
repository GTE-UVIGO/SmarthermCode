#!/bin/bash
#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
# Procesar argumentos:
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		-id|--inidate)
			inidate=$2
			shift 2
			;;
		-fd|--findate)
			findate=$2
			shift 2
			;;
		*)
			echo "GXSXX ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
			exit 101
	esac
done
# Comprobar argumentos:
if [ -z "$inidate" ]; then
	echo "GXSXX ERROR: No se ha proporcionado la variable obligatoria 'inidate'. Operacion cancelada."
	exit 101
fi
if [ -z "$findate" ]; then
	echo "GXSXX ERROR: No se ha proporcionado la variable obligatoria 'findate'. Operacion cancelada."
	exit 101
fi
#%%%%%%%%%%%%%%%%%%%%%%%%%
#% Generar lista de dias %
#%%%%%%%%%%%%%%%%%%%%%%%%%
daylist=()
d="$inidate"
while [[ ! "$d" > "$findate" ]]
do
	daylist+=($d)
	d=$(date -I -d "$d + 1 day")
done
echo ${daylist[@]}