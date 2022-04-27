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
		-d|--date)
			date=$2
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
			echo "GXS02 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
			exit 101
	esac
done
# Comprobar argumentos:
if [ -z "$date" ]; then
	echo "GXS02 ERROR: No se ha proporcionado la variable obligatoria 'date'. Operacion cancelada."
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
taskmes="GXS02_GribFiles_Downloader. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
# Directorio para dia de datos:
DIRECTORYDATE=$(date --date="${date}" +"%Y_%m_%d")
PATHDATE=$(date --date="${date}" +"%Y/%Y%m")
DATE=$(date --date="${date}" +"%Y%m%d")
DIR_Date=$DIR_G2F/$DIRECTORYDATE
mkdir -p $DIR_Date
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Comprobar espacio disponible %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reqspace=$((${#exelist[@]}*${#predlist[@]}*$maxgribsize/1024))
avaspace=$(df -Pk $DIR_Date | awk 'NR==2 {print $4}')
if [ $reqspace -gt $avaspace ]; then
	erromes="GXS02 $modeltxt ERROR: No hay suficiente espacio disponible: $reqspace KB necesarios para $numfil descargas."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
#%%%%%%%%%%%%%%%%%%%%%%
#% Descargar archivos %
#%%%%%%%%%%%%%%%%%%%%%%
filelist=()
nfilelist=()
ercode=0
case "$model$submodel$source" in
	"GFSsfluxNOAA")
		URLmain="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
		# Descargar archivos:
		for exe in ${exelist[@]}
		do
			for pred in ${predlist[@]}
			do
				URLfile="gfs.$DATE/$exe/"
				file="gfs.t"$exe"z.sfluxgrbf"$pred".grib2"
				nfile="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
				filelist+=($file)
				nfilelist+=($nfile)
				wget -P "$DIR_Date" -e robots=off -q -A ".grib2" $URLmain$URLfile$file || ercode=$? &
			done
			sleep 10
		done
		;;
	"GFS0p25NOAA")
		URLmain="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
		# Descargar archivos:	
		for exe in ${exelist[@]}
		do
			for pred in ${predlist[@]}
			do
				URLfile="gfs.$DATE/$exe/"
				file="gfs.t"$exe"z.pgrb2.0p25.f"$pred
				nfile="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
				filelist+=($file)
				nfilelist+=($nfile)
				wget -P "$DIR_Date" -e robots=off -q -A ".f$pred" $URLmain$URLfile$file || ercode=$? &
			done
			sleep 10
		done
		;;
	"GFS0p50NOAA")
		URLmain="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
		# Descargar archivos:	
		for exe in ${exelist[@]}
		do
			for pred in ${predlist[@]}
			do
				URLfile="gfs.$DATE/$exe/"
				file="gfs.t"$exe"z.pgrb2.0p50.f"$pred
				nfile="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
				filelist+=($file)
				nfilelist+=($nfile)
				wget -P "$DIR_Date" -e robots=off -q -A ".f$pred" $URLmain$URLfile$file || ercode=$? &
			done
			sleep 10
		done
		;;
	"GDASsfluxNOAA")
		URLmain="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
		# Descargar archivos:
		for exe in ${exelist[@]}
		do
			for pred in ${predlist[@]}
			do
				URLfile="gdas.$DATE/$exe/"
				file="gdas.t"$exe"z.sfluxgrbf"$pred".grib2"
				nfile="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
				filelist+=($file)
				nfilelist+=($nfile)
				wget -P "$DIR_Date" -e robots=off -q -A ".grib2" $URLmain$URLfile$file || ercode=$? &
			done
			sleep 10
		done
		;;	
	"GDASsfluxRDA")
		# Crear cookie para conectarse al servidor:
		wget --save-cookies "$DIR_Base/cookieRDA" "https://rda.ucar.edu/cgi-bin/login"
		rm "login"
		# Descargar archivos:
		URLmain="https://rda.ucar.edu/data/ds084.4/"
		for exe in ${exelist[@]}
		do
			for pred in ${predlist[@]}
			do
				URLfile="$PATHDATE/"
				file="gdas1.sflux."$DATE$exe".f"$pred".grib2"
				nfile="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
				filelist+=($file)
				nfilelist+=($nfile)
				wget -P "$DIR_Date" -q --load-cookies "$DIR_Base/cookieRDA" $URLmain$URLfile$file || ercode=$? &
			done
			sleep 10
		done
		# Borrar cookie:
		rm "$DIR_Base/cookieRDA"
		;;
	*)
		erromes="GXS02 $modeltxt ERROR: Se ha solicitado una combinacion de modo de ejecucion y fuente de datos no valida: '$Mode' '$source'. Operacion cancelada."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
		;;
esac
wait
if [ ! $ercode -eq 0 ]; then
	echo "merda"
	erromes="GXS02 $modeltxt ERROR: El codigo para descargar archivos ha fallado: Codigo $ercode."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
	exit 101
fi
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Comprobar tamanho de archivos %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for file in ${filelist[@]}
do	
	gribsize=$(wc -c <$DIR_Date/$file); ercode=$?
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS02 $modeltxt ERROR: El codigo para comprobar archivos ha fallado: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	if [ $gribsize -gt $maxgribsize ] || [ $gribsize -lt $mingribsize ]; then
		warnmes="GXS02 $modeltxt AVISO: El tamanho del fichero '$file' queda fuera de los limites esperados: $gribsize bytes."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$warnmes" -mt "W" -ml $lev -lf $logfile -of "$optfile"
	fi
done
#%%%%%%%%%%%%%%%%%%%%%%
#% Renombrar archivos %
#%%%%%%%%%%%%%%%%%%%%%%
k=0
for file in ${filelist[@]}
do	
	mv $DIR_Date/$file $DIR_Date/${nfilelist[$k]}; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS02 $modeltxt ERROR: El codigo para renombrar archivos ha fallado: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	((k++))
done
taskmes="GXS02_GribFiles_Downloader. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
