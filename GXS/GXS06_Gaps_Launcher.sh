#!/bin/bash

#$$$$$$$$$$$$$$
#$ Argumentos $
#$$$$$$$$$$$$$$
# Valores por defecto:
localstored=true
optfile=""
logfile="@dailylog"
newexe=true
lev=0
# Procesar argumentos:
while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
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
			echo "GXS06 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
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
taskmes="GXS06_Gaps_Launcher. Modelo: $model $submodel."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Control de fechas limite %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startdate=$(date -I -d "${enddate} -${ndays_gaps} day +1 day" )
if [[ "$limitstart" > "$startdate" ]]; then 
	warnmes="GXS06 $model $submodel AVISO: El inicio del intervalo solicitado '$startdate' es menor que el limite de inicio de datos '$limitstart'. Intervalo recortado."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	startdate="$limitstart"
fi
if [[ "$limitend" < "$enddate" ]]; then
	warnmes="GXS06 $model $submodel AVISO: El final del intervalo solicitado '$enddate' es mayor que el limite de final de datos '$limitend'. Intervalo recortado."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	enddate=$limitend
fi
if [[ "$enddate" < "$startdate" ]]; then
	warnmes="GXS06 $model $submodel ERROR: El final del intervalo solicitado '$enddate' es menor que el inicio del mismo '$startdate'. Operacion cancelada."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
ndays_gaps=$[($(date -d "$enddate" +"%s")-$(date -d "$startdate" +"%s"))/(24*60*60)+1]
#%%%%%%%%%%%%%%%%%%%%%%%%%
#% Generar lista de dias %
#%%%%%%%%%%%%%%%%%%%%%%%%%
# Llamar a script para analizar base de datos:
infomes="Dias a procesar: $ndays_gaps dias hasta el $enddate."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$infomes" -mt "I" -ml $lev -lf $logfile -of "$optfile"
res=$(Rscript $DIR_Base/GXS07_Gaps_GetErrorDays.R -of "$optfile" -d $enddate -n $ndays_gaps -ct $cortag -it $igntag -lf $logfile -l $lev)
# Interpretar resultados:
outputs=$(echo "$res" | sed -n "s/^.*\(\[O\].*\[O\]\).*$/\1/p")
rescode=$(echo "$res" | sed -n "s/^.*\(\[R\].*\[R\]\).*$/\1/p")
messages="${res//$outputs/""}"
messages="${messages//$rescode/""}"
messages=${messages%?}
outputs="${outputs//"[O] "/""}"
outputs="${outputs//" [O]"/""}"
rescode="${rescode//"[R]: "/""}"
rescode="${rescode//" [R]"/""}"
# TODO: print captured messages on the appropiate logfile.
if [ ! $rescode = 0 ]; then
	if [ $rescode = "" ]; then
		erromes="GXS06 $model $submodel ERROR: No se ha recibido codigo de finalizacion o error al terminar GXS07_Gaps_GetErrorDays."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"	
	elif [ $rescode = 101 ]; then
		erromes="GXS06 $model $submodel ERROR: Se ha recibido codigo de error controlado al ejecutar GXS07_Gaps_GetErrorDays."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	elif [ $rescode = 102 ]; then
		erromes="GXS06 $model $submodel ERROR: Se ha recibido codigo de error en argumentos al ejecutar GXS07_Gaps_GetErrorDays."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	fi
	exit 101
fi
# Enviar lista a script para descargar e insertar nuevos archivos:
daylist=($outputs)
if [ ! ${#daylist[@]} -eq 0 ]; then
	warnmes="GXS06 $model $submodel AVISO: Se han encontrado dias con errores. Reparando..."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	$DIR_Base/GXS00_GribFiles_Launcher.sh -dl "$(echo ${daylist[@]})" -ls $localstored -of "$optfile" -lf $logfile -ne false -l $lev; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		if [ ! $ercode -eq 101 ]; then
			erromes="GXS06 $model $submodel ERROR: Error no controlado en GXS00_GribFiles_Launcher: $errcode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"	
		fi
	exit 101
fi
fi
taskmes="GXS06_Gaps_Launcher. Modelo: $model $submodel."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
