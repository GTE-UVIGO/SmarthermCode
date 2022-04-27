#!/bin/bash

#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
# Valores por defecto:
localstored=false
optfile=""
logfile="@dailylog"
newexe=true
lev=0
# Procesar argumentos:
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		-dl|--daylist)
			daylist=($2)
			shift 2
			;;
		-ls|--localstored)
			localstored=($2)
			shift 2
			;;
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
			echo "GXS03 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
			exit 101
	esac
done
# Comprobar argumentos:
if [ -z "$daylist" ]; then
	echo "GXS03 ERROR: No se ha proporcionado la variable obligatoria 'daylist'. Operacion cancelada."
	exit 101
fi
#$$$$$$$$$$$$$
#$ Variables $
#$$$$$$$$$$$$$
# Cargar variables globales:
source $optfile
# Comprobar directorio de ficheros de registro:
mkdir -p $DIR_Log

modeltxt="$model $submodel $suffix"
modeltxt="$(echo -e "${modeltxt}" | sed -e 's/[[:space:]]*$//')"
taskmes="GXS03_GribFiles_ToServer. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%
#% Iterar dias %
#%%%%%%%%%%%%%%%
for d in ${daylist[@]}
do
	ndesrec=$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)
	# Directorio para cada dia de datos:
	DIRECTORYDATE=$(date --date="${d}" +"%Y_%m_%d")
	DATE=$(date --date="${d}" +"%Y%m%d")
	DIR_Date=$DIR_G2F/$DIRECTORYDATE
	#%%%%%%%%%%%%%%%%%%
	#% Transferir TAR %
	#%%%%%%%%%%%%%%%%%%
	mkdir -p $DIR_Date
	Grib2Folder=$(echo $DIR_G2F | rev | cut -d'/' -f 1 | rev)
	if [ ! $localstored = true ]; then
		rsync -e -rp --chmod=ug+rwx --timeout=$CESGAtimeout $Grib2Folder/$DIRECTORYDATE $DIR_G2F; ercode=$?
		sleep 2
		if [ ! $ercode -eq 0 ]; then
			erromes="GXS03 $modeltxt ERROR*: El codigo para transferir archivos desde el CESGA ha fallado: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
			continue
		fi
	fi
	#%%%%%%%%%%%%%%%%%%%%
	#% Descomprimir TAR %
	#%%%%%%%%%%%%%%%%%%%%
	cd $DIR_Date
	filenameTAR="$model$submodel$suffix$DATE.tar.gz"
	tar -xzf $filenameTAR; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS03 $modeltxt ERROR*: El codigo para descomprimir archivos ha fallado: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		continue
	fi
	ndesrec=$(($(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)-ndesrec))
	cd $DIR_Base
	#%%%%%%%%%%%%%%%%%%
	#% Mover archivos %
	#%%%%%%%%%%%%%%%%%%
	for file in $DIR_Date/*.grib2
	do
		mv $file $DIR_G2F; ercode=$?
		if [ ! $ercode -eq 0 ]; then
			erromes="GXS03 $modeltxt ERROR*: El codigo para mover archivos ha fallado: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
			continue
		fi
	done
	#%%%%%%%%%%%%%%
	#% Borrar TAR %
	#%%%%%%%%%%%%%%
	recerror=false
	rm $DIR_Date/$filenameTAR || ( recerror="true" && ercode=$? )
	rmdir $DIR_Date || ( recerror="true" && ercode=$? )
	if [ $recerror = "true" ]; then
		erromes="GXS03 $modeltxt ERROR*: El codigo para borrar archivos comprimidos ha fallado: $ercode (U)."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		continue
	fi
	#%%%%%%%%%%%%%%%%%%%%%%%%
	#% Resumen de descargas %
	#%%%%%%%%%%%%%%%%%%%%%%%%
	infomes="Dia $d: $ndesrec archivos descargados."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$infomes" -mt "I" -ml $lev -lf $logfile -of "$optfile"
done
taskmes="GXS03_GribFiles_ToServer. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
