#!/bin/bash

#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
# Valores por defecto:
optfile=""
logfile="@dailylog"
newexe=true
lev=0
# Procesar argumentos:
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		-of|--optfile)
			optfile=$2
			shift 2
			;;
		-lf|--logfile)
			logfile=$2
			shift 2
			;;
		-ne|--newexe)
			newexe=$2
			shift 2
			;;
		-l|--lev)
			lev=$(($2+1))
			shift 2
			;;
		*)
			echo "GXS04 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
			exit 101
	esac
done
#$$$$$$$$$$$$$
#$ Variables $
#$$$$$$$$$$$$$
# Cargar variables globales:
source $optfile
# Comprobar directorio de ficheros de registro:
mkdir -p $DIR_Log

modeltxt="$model $submodel $suffix"
modeltxt="$(echo -e "${modeltxt}" | sed -e 's/[[:space:]]*$//')"
taskmes="GXS04_GribFiles_ToDataBase. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%
#% Iterar archivos %
#%%%%%%%%%%%%%%%%%%%
# Comprobar si hay archivos que iterar:
if [ $(ls $DIR_G2F/*.grib2 2>/dev/null | wc -l) -eq 0 ]; then
	erromes="GXS04 $modeltxt ERROR: No se ha encontrado ningun archivo grib2."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
niterec=0
nerrorec=0
for file in $DIR_G2F/*.grib2
do
	# Leer codigo de prediccion:
	strRootName=$(echo $file | rev | cut -d'/' -f 1 | rev)
	pred=$(echo $strRootName | cut -d'.' -f 6)
	gridname=$(echo $strRootName | cut -d'.' -f 4)
	# Llamar a script para procesar archivo:
	res=$(Rscript $DIR_Base/GXS05_GribFiles_ToDataBase.R -bf $file -p $pred -gn $gridname -ct $cortag -it $igntag -of "$optfile" -lf $logfile -l $lev)
	# Extraer codigo de operacion y mensajes:
	rescode=$(echo "$res" | sed -n "s/^.*\(\[R\].*\[R\]\).*$/\1/p")
	messages="${res//$outputs/""}"
	messages="${messages//$rescode/""}"
	messages=${messages%?}
	rescode="${rescode//"[R]: "/""}"
	rescode="${rescode//" [R]"/""}"
	# Procesar codigo de operacion:
	if [ ! $rescode = 0 ]; then
		if [ $rescode = "" ]; then
			erromes="GXS04 $modeltxt ERROR*: No se ha recibido codigo de finalizacion o error al terminar GXS05_GribFiles_ToDataBase con el archivo '$file'."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		elif [ $rescode = 101 ]; then
			erromes="GXS04 $modeltxt ERROR*: Se ha recibido codigo de error controlado al ejecutar GXS05_GribFiles_ToDataBase con el archivo '$file'."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		elif [ $rescode = 102 ]; then
			erromes="GXS04 $modeltxt ERROR*: Se ha recibido codigo de error en argumentos al ejecutar GXS05_GribFiles_ToDataBase con el archivo '$file'."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		fi
		((nerrorec++))
	fi
	((niterec++))
done
# Resumen de archivos iterados:
infomes="Total: $niterec archivos iterados. Errores: $nerrorec."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$infomes" -mt "I" -ml $lev -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%
#% Borrar archivos %
#%%%%%%%%%%%%%%%%%%%
rm -rf $DIR_G2F; ercode=$?
if [ ! $ercode -eq 0 ]; then
	erromes="GXS04 $modeltxt ERROR: El codigo para borrar archivos ha fallado: Codigo $ercode."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
if [ -f "$DIR_Base/my.inv" ]; then rm "$DIR_Base/my.inv"; fi
if [ -f "$HOME/my.inv" ]; then rm "$HOME/my.inv"; fi
taskmes="GXS04_GribFiles_ToDataBase. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
