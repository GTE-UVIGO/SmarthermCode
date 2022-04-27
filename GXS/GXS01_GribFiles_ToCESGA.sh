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
			echo "GXS01 ERROR: Se ha proporcionado una flag '$1' no reconocida. Operacion cancelada."
			exit 101
	esac
done
# Comprobar argumentos:
if [ -z "$daylist" ]; then
	echo "GXS01 ERROR: No se ha proporcionado la variable obligatoria 'daylist'. Operacion cancelada."
	exit 101
fi
#$$$$$$$$$$$$$
#$ Variables $
#$$$$$$$$$$$$$
# Cargar variables globales:
source $optfile
# Comprobar directorio de ficheros de registro:
mkdir -p $DIR_Log
# Comprobar directorio de almacen local:
mkdir -p $DIR_Store

modeltxt="$model $submodel $suffix"
modeltxt="$(echo -e "${modeltxt}" | sed -e 's/[[:space:]]*$//')"
taskmes="GXS01_GribFiles_ToCESGA. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mn $newexe -mi "Ini" -lf $logfile -of "$optfile"
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Iterar dias, ejecuciones, predicciones %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for d in ${daylist[@]}
do
	# Directorio para cada dia de datos:
	DIRECTORYDATE=$(date --date="${d}" +"%Y_%m_%d")
	PATHDATE=$(date --date="${d}" +"%Y/%Y%m")
	DATE=$(date --date="${d}" +"%Y%m%d")
	DIR_Date=$DIR_G2F/$DIRECTORYDATE
	if [ -d $DIR_Date ]; then rm -rf $DIR_Date; fi
	mkdir -p $DIR_Date
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Descargar archivos originales %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	ndesori=$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)
	# Descargar archivos:
	$DIR_Base/GXS02_GribFiles_Downloader.sh -d $d -of "$optfile" -lf $logfile -ne false -l $lev; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		if [ ! $ercode -eq 101 ]; then
			erromes="GXS01 $modeltxt ERROR: Error no controlado en GXS02_GribFiles_Downloader: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		fi
		exit 101
	fi	
	# Crear lista de archivos (renombrados):
	filelist=()
	for exe in ${exelist[@]}
	do
		for pred in ${predlist[@]}
		do
			file="$model.$submodel$suffix.$DATE.$exe.$pred.grib2"
			filelist+=($file)
		done
	done
	ndesori=$(($(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)-ndesori))
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Recortar archivos originales %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	nrecori=$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)
	recerror=false
	for file in ${filelist[@]}
	do
		# Recortes para Peninsula Iberica:
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -10.5:-3.5 34.5:45 $DIR_Date/${file/$DATE/$DATE.IB1} 1>/dev/null || ercode=$?
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -3.5:-0.5 34:44.5 $DIR_Date/${file/$DATE/$DATE.IB2} 1>/dev/null || ercode=$?
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -0.5:2.5 36.5:44 $DIR_Date/${file/$DATE/$DATE.IB3} 1>/dev/null || ercode=$?
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib 2.5:5.5 38:43.5 $DIR_Date/${file/$DATE/$DATE.IB4} 1>/dev/null || ercode=$?
		# Uniones para Peninsula Iberica:
		cat $DIR_Date/${file/$DATE/$DATE.IB1} $DIR_Date/${file/$DATE/$DATE.IB2} $DIR_Date/${file/$DATE/$DATE.IB3} $DIR_Date/${file/$DATE/$DATE.IB4} > $DIR_Date/${file/$DATE/$DATE.IBX} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.IB1} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.IB2} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.IB3} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.IB4} || ercode=$?
		# Recortes para Macaronesia:
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -32.5:-24 35.5:41 $DIR_Date/${file/$DATE/$DATE.MC1} 1>/dev/null || ercode=$?
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -18.5:-15 31:34.5 $DIR_Date/${file/$DATE/$DATE.MC2} 1>/dev/null || ercode=$?
		/usr/local/bin/wgrib2 -v0 $DIR_Date/$file -small_grib -19.5:-12 26.5:31 $DIR_Date/${file/$DATE/$DATE.MC3} 1>/dev/null || ercode=$?
		# Uniones para Macaronesia:
		cat $DIR_Date/${file/$DATE/$DATE.MC1} $DIR_Date/${file/$DATE/$DATE.MC2} $DIR_Date/${file/$DATE/$DATE.MC3} > $DIR_Date/${file/$DATE/$DATE.MCX} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.MC1} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.MC2} || ercode=$?
		rm $DIR_Date/${file/$DATE/$DATE.MC3} || ercode=$?
	done
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS01 $modeltxt ERROR: El codigo para recortar archivos originales ha fallado: $ercode (U)."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	nrecori=$(($(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)-nrecori))
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Borrar archivos originales %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	nborori=$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)
	for file in ${filelist[@]}	
	do
		rm $DIR_Date/$file; ercode=$?
		if [ ! $ercode -eq 0 ]; then
			erromes="GXS01 $modeltxt ERROR: El codigo para borrar archivos originales ha fallado: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
			exit 101
		fi
	done
	wait
	nborori=$((nborori-$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)))
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Comprimir archivos recortados %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	cd $DIR_Date
	filenameTAR="$model$submodel$suffix$DATE.tar.gz"
	tar -czf $filenameTAR *; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS01 $modeltxt ERROR: El codigo para comprimir archivos recortados ha fallado: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	cd $DIR_Base
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Borrar archivos recortados %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	nborrec=$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)
	find $DIR_Date -name "*.grib2" -delete; ercode=$?
	if [ ! $ercode -eq 0 ]; then
		erromes="GXS01 $modeltxt ERROR: El codigo para borrar archivos recortados ha fallado: Codigo $ercode."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	nborrec=$((nborrec-$(ls $DIR_Date/*.grib2 2>/dev/null | wc -l)))
	chmod -R 777 $DIR_G2F
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Resumen de descargas y recortes %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	infomes="Dia $d: $ndesori archivos descargados | $nrecori archivos recortados."
	Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$infomes" -mt "I" -ml $lev -lf $logfile -of "$optfile"
	if [ ! $ndesori -eq $nborori ] || [ ! $nrecori -eq $nborrec ]; then
		erromes="GXS01 $modeltxt ERROR: El numero de archivos generados y eliminados no coincide."
		Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
		exit 101
	fi
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Transferir archivos comprimidos al CESGA %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if [ ! $saveCESGA = false ]; then
		Grib2Folder=$(echo $DIR_G2F | rev | cut -d'/' -f 1 | rev)
		rsync -e  -rp --chmod=ug+rwx --timeout=$CESGAtimeout $DIR_Date "gtesmart/$Grib2Folder"; ercode=$?
		sleep 2
		if [ ! $ercode -eq 0 ]; then
			erromes="GXS01 $modeltxt ERROR: El codigo para transferir archivos al CESGA ha fallado: Codigo $ercode."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
			exit 101
		fi
	fi
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#% Copiar archivos comprimidos al almacen local %
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if [ ! $savelocal = false ]; then
		recerror=false
		mkdir -p $DIR_Store/$DIRECTORYDATE
		cp $DIR_Date/$filenameTAR $DIR_Store/$DIRECTORYDATE || ercode=$?
		chmod -R 777 $DIR_Store || ercode=$?
		if [ ! $localstored = true ]; then
			rm -rf $DIR_G2F || ercode=$?
		fi
		if [ ! $ercode -eq 0 ]; then
			erromes="GXS01 $modeltxt ERROR: El codigo para copiar archivos al almacen local ha fallado: $ercode (U)."
			Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$erromes" -mt "E" -ml $lev -lf $logfile -of "$optfile"
			exit 101
		fi
	fi
done
taskmes="GXS01_GribFiles_ToCESGA. Modelo: $modeltxt."
Rscript $DIR_Base/GXSXX_Tools_LogWrite.R -m "$taskmes" -mt "T" -ml $lev -mi "Fin" -lf $logfile -of "$optfile"
