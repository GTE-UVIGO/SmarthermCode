#!/bin/bash

#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
# Valores por defecto:
givenlist=false
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
			givenlist=true
			shift 2
			;;
		-ls|--localstored)
			localstored=$2
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
			echo "GXS00 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
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
taskmes="GXS00_GribFiles_Launcher. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Control de fechas limite %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startdate=$(date -I -d "${enddate} -${ndays_grib} day +1 day" )
if [[ "$limitstart" > "$startdate" ]]; then 
	warnmes="GXS00 $model $submodel AVISO: El inicio del intervalo solicitado '$startdate' es menor que el limite de inicio de datos '$limitstart'. Intervalo recortado."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	startdate="$limitstart"
fi
if [[ "$limitend" < "$enddate" ]]; then
	warnmes="GXS00 $model $submodel AVISO: El final del intervalo solicitado '$enddate' es mayor que el limite de final de datos '$limitend'. Intervalo recortado."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	enddate=$limitend
fi
if [[ "$enddate" < "$startdate" ]]; then
	erromes="GXS00 $modeltxt ERROR: El final del intervalo solicitado '$enddate' es menor que el inicio del mismo '$startdate'. Operacion cancelada."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
#%%%%%%%%%%%%%%%%%%%%%%%%%
#% Generar lista de dias %
#%%%%%%%%%%%%%%%%%%%%%%%%%
if [ $givenlist = false ]; then
	daylist=()
	d="$startdate"
	while [[ ! "$d" > "$enddate" ]]
	do
		daylist+=($d)
		d=$(date -I -d "$d + 1 day")
	done
fi
infomes="Dias a procesar: ${daylist[@]}."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$infomes" -mt "I" -ml $lev -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Descargar archivos al CESGA %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$DIR_Base/GXS01_GribFiles_ToCESGA.sh -dl "$(echo ${daylist[@]})" -ls $localstored -of "$optfile" -lf $logfile -ne false -l $lev; ercode=$?
if [ ! $ercode -eq 0 ]; then
	if [ ! $ercode -eq 101 ]; then
		erromes="GXS00 $modeltxt ERROR: Error no controlado en GXS01_GribFiles_ToCESGA: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"	
	fi
	exit 101
fi
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Descargar archivos al Server %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if [ ! $insertdb = false ]; then
	$DIR_Base/GXS03_GribFiles_ToServer.sh -dl "$(echo ${daylist[@]})" -ls $localstored -of "$optfile" -lf $logfile -ne false -l $lev; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		if [ ! $ercode -eq 101 ]; then
			erromes="GXS00 $modeltxt ERROR: Error no controlado en GXS03_GribFiles_ToServer: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"	
		fi
		exit 101
	fi
fi
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Insertar en Base de Datos %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if [ ! $insertdb = false ]; then
	$DIR_Base/GXS04_GribFiles_ToDataBase.sh -of "$optfile" -lf $logfile -ne false -l $lev; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		if [ ! $ercode -eq 101 ]; then
			erromes="GXS00 $modeltxt ERROR: Error no controlado en GXS04_GribFiles_ToDataBase: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"	
		fi
		exit 101
	fi
fi
taskmes="GXS00_GribFiles_Launcher. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
