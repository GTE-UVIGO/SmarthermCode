#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_AddPoints %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_AddPoints = function(grib2file,pred,scriptPath,optfile,logfile="@console",lev=0){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca e inserta nuevos puntos en la base de datos.                                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] grib2file: Nombre del fichero grib2 procesado (string).                              %
  #% [I] pred: Hora de prediccion (string).                                                   %
  #% [I] scriptPath: Ruta absoluta de ubicacion de scripts (string).                          %
  #% [I] optfile: Nombre del fichero de opciones globales (string).                           %
  #% [I] logfile: Nombre del fichero donde se escribe el mensaje (string) (opcional).         %
  #% [I] lev: Nivel de impresion de mensaje (integer) (opcional).                         	  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] GXSXX_Tools_LogWrite.R: Imprime mensajes en un fichero de texto.                     %
  #% [S] GXSXX_RTools_CommonTools.R: Contiene funciones y variables comunes para ficheros R.  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Codigo de validacion de ejecucion.                                           %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Cargar scripts necesarios:
  source(paste0(scriptPath,"/GXSXX_Tools_LogWrite.R"))
  source(paste0(scriptPath,"/GXSXX_RTools_CommonTools.R"))
  # Cargar variables globales:
  dbConnection = GXSXX_OpenConnection(optfile)
  dbName = GXSXX_ReadOptions("dbName",optfile)
  dbTable_Elevation = GXSXX_ReadOptions("dbTable_Elevation",optfile)
  libpath = GXSXX_ReadOptions("libpath",optfile) 
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("DBI")){require("DBI",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}			# Para trabajar con bases de datos
	if (GXSXX_CheckPack("RMySQL")){require("RMySQL",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para conectar con MySQL
	if (GXSXX_CheckPack("httr")){require("httr",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}			# Por conflictos con rNOMADS en GXSXX_Tools_LogWrite
	if (GXSXX_CheckPack("rNOMADS")){require("rNOMADS",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}	# Para trabajar con ficheros grib2
	if (GXSXX_CheckPack("dplyr")){require("dplyr",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para trabajar con dataframes
	if (GXSXX_CheckPack("rjson")){require("rjson",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para procesar archivos JSON
  })
  # Mensajes de inicio de tarea:
  mText = paste0("GXSXX_BTools_AddPoints.")
  GXSXX_LogWrite(mText=mText,mType="T",mNewEx=TRUE,mLevel=lev,mIniFin="Ini",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  mText = paste0("Archivo recibido: '",basename(grib2file),"'.")
  GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Leer fichero de coordenadas %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  numpred = as.numeric(pred)
  levelsSur = c("surface")
  timesAnl = c("anl")
  timesfhf = c(paste0(numpred," hour fcst"))
  if (identical(numpred,0)){timesIns = timesAnl} else {timesIns = timesfhf}
  dHGT = as.data.frame(ReadGrib(grib2file,levelsSur,"HGT",timesIns),stringsAsFactors=FALSE)
  coords = dHGT[,names(dHGT) %in% c("lon","lat")]
  fulllist = round(coords,4)
  # Trocear y procesar lista de puntos:
  limpoints = 25000
  nchunks = ceiling(nrow(fulllist)/limpoints)
  splitlist = split(fulllist,rep(1:nchunks,each=limpoints)[1:nrow(fulllist)])
  for (i in 1:nchunks){
    coordslist = splitlist[[i]]
    mText = paste0("Procesando chunck ",i," de ",nchunks,".")
    GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
    #%%%%%%%%%%%%%%%%%%%%
    #% Comprobar puntos %
    #%%%%%%%%%%%%%%%%%%%%
	# Comprobar lista vacia y preparar formato de numeros:
    if (nrow(coordslist)==0){
      mText = paste0("GXSXX ERROR: Se ha recibido un listado sin ningun punto.")
      GXSXX_LogWrite(mText=mText,mType="E",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
	  return("[R]: 101")
    } else {
      coordslist$lat = formatC(coordslist$lat,digits=4,format="f")
      coordslist$lon = formatC(coordslist$lon,digits=4,format="f")
      coordslist$lat[coordslist$lat=="-0.0000"] = "0.0000"
      coordslist$lon[coordslist$lon=="-0.0000"] = "0.0000"
    }
    # Comprobar lista de puntos:
    checkpointslist = GXSXX_CheckPoints(coordslist,dbConnection,dbName,dbTable_Elevation)
    if (!nrow(checkpointslist)==0){
      mText = paste0("GXSXX AVISO: Se han encontrado ",nrow(checkpointslist)," puntos en el archivo grib que no existen en la base de datos.")
      GXSXX_LogWrite(mText=mText,mType="W",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
	  #%%%%%%%%%%%%%%%%%%%%%%%%%%
      #% Insertar nuevos puntos %
      #%%%%%%%%%%%%%%%%%%%%%%%%%%
      nnp = GXSXX_InsertPoints(checkpointslist,dbConnection,dbName,dbTable_Elevation)
      mText = paste0("Nuevos puntos insertados: ",nnp,".")
      GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
    }
  }
  # Terminar, cerrar conexion y devolver codigo de ejecucion correcta:
  dbDisconnect(dbConnection)
  mText = paste0("GXSXX_BTools_AddPoints.")
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Fin",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  return("[R]: 0")
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_CheckPoints %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_CheckPoints = function(coordslist,dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca puntos no existentes en la base de datos.                                          %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] coordslist: Lista de datos lon, lat 4f (dataframe).                                  %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: lista de datos lat, lon 4f (dataframe) (NULL: sin resultados).               %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de valores:
  coords = toString(paste0("(",paste(coordslist$lat,coordslist$lon,sep=","),")"))
  # Preparar cadenas de consultas:
  strQuery1 = paste("CREATE TEMPORARY TABLE temp_Gridtable (`latitud` decimal(7,4),`longitud` decimal(7,4));",collapse=" ")
  strQuery2 = paste("INSERT INTO temp_Gridtable (latitud, longitud) VALUES",coords,";",collapse=" ")
  strQuery3 = paste("SELECT PuntosGrib.latitud AS lat, PuntosGrib.longitud AS lon
                    FROM
                      (SELECT * FROM temp_Gridtable) AS PuntosGrib
                    LEFT OUTER JOIN
                      (SELECT idGrid, latitud, longitud FROM ",paste0(dbName,".",dbTable_Elevation),") AS PuntosBase
                    ON (PuntosGrib.latitud = PuntosBase.latitud AND PuntosGrib.longitud = PuntosBase.longitud)
                    WHERE PuntosBase.idGrid IS NULL;",collapse=" ")
  strQuery4 = paste("DROP TABLE temp_Gridtable;",collapse=" ")
  # Lanzar consultas:
  suppressWarnings({
    stmres1 = dbSendStatement(dbConnection, strQuery1)
    stmres2 = dbSendStatement(dbConnection, strQuery2)
    d = dbGetQuery(dbConnection, strQuery3)
    stmres4 = dbSendStatement(dbConnection, strQuery4)
    dbClearResult(stmres1)
    dbClearResult(stmres2)
    dbClearResult(stmres4)
  })
  # Ajustar formato de resultados:
  if (! nrow(d)==0){
    d$lat = formatC(d$lat,digits=4,format="f")
    d$lon = formatC(d$lon,digits=4,format="f")
  }
  # Terminar y devolver resultados:
  results = d
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_InsertPoints %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_InsertPoints = function(coordslist,dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Inserta puntos en la base de datos.                                                      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] coordslist: lista de datos lon, lat 4f (dataframe).                             	  %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: numero de puntos insertados (numeric) (NULL: sin resultados).                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de valores:
  lat = coordslist$lat
  lon = coordslist$lon
  coords = paste(toString(paste0("(",paste(lat,lon,sep=","),")")),sep=",")
  # Preparar cadena de consulta:
  strQuery = paste("INSERT INTO",paste0(dbName,".",dbTable_Elevation),"(latitud,longitud)",
                   "VALUES",coords,"ON DUPLICATE KEY UPDATE idGrid=idGrid;",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({
    stmres = dbSendStatement(dbConnection, strQuery)
    dbClearResult(stmres)
  })
  # Terminar:
  results = length(coordslist$lon)
  if (is.null(results)) return(NULL) else return(results)
}
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$ Funcion de error controlado $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
rSQL_ContrError = function(){return("[R]: 101")}
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$ Funcion de error en argumentos $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
rSQL_ArgError = function(){return("[R]: 102")}
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$ Lanzar funcion desde Bash $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Determinar directorio de ejecucion:
match = grep("--file=",commandArgs(trailingOnly=FALSE))
scriptPath = dirname(normalizePath(sub("--file=","",commandArgs(trailingOnly=FALSE)[match])))
# Valores por defecto:
logfile = "@console"
lev = 0
# Procesar argumentos de funcion:
args = commandArgs(trailingOnly = TRUE)
while (length(args) > 0){
  key = args[1]
  if (key=="-of"|key=="--optfile"){
    optfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-bf"|key=="--grib2file"){
    grib2file = args[2];args = args[-1];args = args[-1]
  } else if (key=="-p"|key=="--pred"){
    pred = args[2];args = args[-1];args = args[-1]
  } else if (key=="-lf"|key=="--logfile"){
    logfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-l"|key=="--lev"){
    lev = as.numeric(args[2])+1;args = args[-1];args = args[-1]
  } else {rSQL_ArgError()}
}
# Lanzar funcion principal:
if (exists("optfile") && exists("grib2file") && exists("pred")){
	GXSXX_AddPoints(grib2file,pred,scriptPath,optfile,logfile,lev)
  } else {rSQL_ArgError()}