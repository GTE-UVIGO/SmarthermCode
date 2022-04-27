#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS05_ToDataBase %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS05_ToDataBase = function(grib2file,pred,gridname,cortag,igntag,scriptPath,optfile,logfile="@console",lev=0){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Lee un archivo grib2 e inserta sus datos en una tabla de instantes.                      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] grib2file: Nombre del fichero grib2 procesado (string).                              %
  #% [I] pred: Hora de prediccion (string).                                                   %
  #% [I] gridname: Nombre de malla de puntos (string).                                        %
  #% [I] cortag: Valor del tag para datos correctos (integer).                                %
  #% [I] igntag: Valor del tag para datos ignorados (integer).                                %
  #% [I] scriptPath: Ruta absoluta de ubicacion de scripts (string).                          %
  #% [I] optfile: Nombre del fichero de opciones globales (string).                           %
  #% [I] logfile: Nombre del fichero donde se escribe el mensaje (string) (opcional).         %
  #% [I] lev: Nivel de impresion de mensaje (integer) (opcional).                         	  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] GXSXX_Tools_LogWrite.R: Imprime mensajes en un fichero de texto.                     %
  #% [S] GXSXX_RTools_CommonTools.R: Contiene funciones y variables comunes para ficheros R.  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Codigo de validacion de ejecucion (string).                                  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Cargar scripts necesarios:
  source(paste0(scriptPath,"/GXSXX_Tools_LogWrite.R"))
  source(paste0(scriptPath,"/GXSXX_RTools_CommonTools.R"))
  # Cargar variables globales:
  dbConnection = GXSXX_OpenConnection(optfile)
  dbName = GXSXX_ReadOptions("dbName",optfile)
  dbTable = GXSXX_ReadOptions("dbTable",optfile)
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
  mText = paste0("GXS05_GribFiles_ToDataBase. Archivo: '",basename(grib2file),"'.")
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Ini",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Leer coordenadas e instante %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Definir capas y tiempos de variables:
  numpred = as.numeric(pred)
  levelsSur = c("surface")
  levels02m  = c("2 m above ground")
  levels10m = c("10 m above ground")
  timesAnl = c("anl")
  timesA24 = c("0-0 day ave fcst")
  timesfhf = c(paste0(numpred," hour fcst"))
  timesAff = c(paste0(floor((numpred-1)/6)*6,"-",numpred," hour ave fcst"))
  if (identical(numpred,0)){
    timesIns = timesAnl
    timesAve = timesA24
  } else {
    timesIns = timesfhf
    timesAve = timesAff
  }
  # Leer lista de coordenadas e instante:
  coords = as.data.frame(ReadGrib(grib2file,levelsSur,"HGT",timesIns),stringsAsFactors=FALSE)  
  Instante = coords$forecast.date[1]
  coords = coords[,names(coords) %in% c("lon","lat")]
  # Preparar formato de coordenadas:
  coords = trunc(coords*10^3)/10^3
  #%%%%%%%%%%%%%%%%%%
  #% Obtener puntos %
  #%%%%%%%%%%%%%%%%%%
  # Comprobar lista de coordenadas vacia:
  if (nrow(coords)==0){
	mText = paste0("GXS05 ERROR: Se ha recibido un listado sin ningun punto para el archivo '",basename(grib2file),"'.")
    GXSXX_LogWrite(mText=mText,mType="E",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
    return("[R]: 101")
  }
  # Solicitar lista de puntos:
  pointslist = GXS05_Coords2Points(coords,dbConnection,dbName,dbTable_Elevation)
  # Comprobar puntos no encontrados:
  if (!nrow(pointslist)==nrow(coords)){
	mText = paste0("GXS05 ERROR: Se han encontrado ",nrow(coords)-nrow(pointslist)," puntos en el archivo '",basename(grib2file),"' que no existen en la base de datos.")
    GXSXX_LogWrite(mText=mText,mType="E",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
    return("[R]: 101")
  }
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Activar puntos inactivos %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Comprobar puntos inactivos:
  nonactlist = pointslist[pointslist$act==0,]
  if (!nrow(nonactlist)==0){
    # Activar puntos inactivos:
	mText = paste0("GXS05 AVISO: Se han encontrado ",nrow(nonactlist)," puntos solicitados no activados en el archivo '",basename(grib2file),"'. Activando...")
    GXSXX_LogWrite(mText=mText,mType="W",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
	nonactlist = data.frame("idGrid"=nonactlist$idGrid)
	GXS05_ActivatePoints(nonactlist,gridname,dbConnection,dbName,dbTable_Elevation)
  }  
  #%%%%%%%%%%%%%%%%%%%%%%
  #% Procesar variables %
  #%%%%%%%%%%%%%%%%%%%%%%
  # Leer listas de variables solicitadas:
  dHGT = ReadGrib(grib2file,levelsSur,"HGT",timesIns)$value
  dTMP = ReadGrib(grib2file,levels02m,"TMP",timesIns)$value
  dPRES = ReadGrib(grib2file,levelsSur,"PRES",timesIns)$value
  dSPFH = ReadGrib(grib2file,levels02m,"SPFH",timesIns)$value
  dDSWRF = ReadGrib(grib2file,levelsSur,"DSWRF",timesIns)$value
  dUGRD = ReadGrib(grib2file,levels10m,"UGRD",timesIns)$value
  dVGRD = ReadGrib(grib2file,levels10m,"VGRD",timesIns)$value
  if (identical(numpred,0)){dPRATE = rep(0,times=length(dHGT))}
  else {dPRATE = ReadGrib(grib2file,levelsSur,"PRATE",timesAve)$value}
  # Preparar resultados de HGT:
  HGT = dHGT
  # Preparar resultados de temperatura:
  TA = dTMP-273.15
  # Preparar resultados de presion:
  PR = dPRES
  # Preparar resultados de humedad relativa:
  SPFH = dSPFH
  p1 = PR*SPFH
  cte = 0.622+SPFH
  ex = exp(14.2928-(5291/(TA+273.15)))
  HR = (p1/(cte*100000*ex))*100
  # Preparar resultados de radiacion solar:
  RS = dDSWRF
  # Preparar resultados de velocidad y direccion de viento:
  UGRD = dUGRD
  VGRD = dVGRD
  VV = sqrt(UGRD^2+VGRD^2)
  DV = 270-(atan(VGRD/UGRD)*180)/pi
  DV[UGRD<0] = 90-(atan(VGRD[UGRD<0]/UGRD[UGRD<0])*180)/pi
  DV[is.na(DV)] = 0
  # Preparar resultados de precipitacion:
  PP = dPRATE*3600
  # Preparar dataframe final:
  data = data.frame(coords,Instante,HGT,TA,PR,HR,RS,VV,DV,PP,
					stringsAsFactors=FALSE)
  data = inner_join(data,pointslist,by=c("lat","lon")) 
  #%%%%%%%%%%%%%%%%%%%
  #% Insertar f-hora %
  #%%%%%%%%%%%%%%%%%%%
  # Configuracion de MySQL:
  limpoints = 5000
  # Recortar dataframes de datos:
  nchunks = ceiling(nrow(pointslist)/limpoints)
  if (identical(numpred,0)){tagPP=igntag} else {tagPP=cortag}
  gridvalues = paste0("(",paste(paste0("'",data$Instante,"'"),
				data$idGrid,data$HGT,data$TA,data$HR,data$PR,data$RS,data$VV,data$DV,data$PP,
				cortag,cortag,cortag,cortag,cortag,cortag,tagPP,
				sep=","),")")
  smallvalues = as.vector(split(gridvalues,rep(1:nchunks,each=limpoints)[1:length(gridvalues)]))
  vQueries = matrix(nrow=nchunks,ncol=1)
  # Preparar vector de cadenas de consulta:
  for (i in 1:nchunks){
    values = toString(smallvalues[[i]])
    strQuery = paste("INSERT INTO",paste0(dbName,".",dbTable),"(instante,idGrid,HGT,TA,HR,PR,RS,VV,DV,PP,TA_tg,HR_tg,PR_tg,RS_tg,VV_tg,DV_tg,PP_tg)
					  VALUES",values,"
                      ON DUPLICATE KEY UPDATE
                      HGT=VALUES(HGT), TA=VALUES(TA), HR=VALUES(HR), PR=VALUES(PR), RS=VALUES(RS), VV=VALUES(VV), DV=VALUES(DV), PP=VALUES(PP),
                      TA_tg=VALUES(TA_tg), HR_tg=VALUES(HR_tg), PR_tg=VALUES(PR_tg), RS_tg=VALUES(RS_tg), VV_tg=VALUES(VV_tg), DV_tg=VALUES(DV_tg), PP_tg=VALUES(PP_tg);",collapse=" ")
    vQueries[i] = strQuery}
  # Lanzar conjunto de consultas:
  suppressWarnings({
    for (i in 1:nchunks){
      stmres = dbSendStatement(dbConnection, vQueries[i])
      dbClearResult(stmres)
    }
  })
  # Terminar, cerrar conexion y devolver codigo de ejecucion correcta:
  dbDisconnect(dbConnection)
  mText = paste0("Datos de puntos insertados: ",length(pointslist$idGrid),".")
  GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  mText = paste0("GXS05_GribFiles_ToDataBase. Archivo: '",basename(grib2file),"'.")
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Fin",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  return("[R]: 0 [R]")
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS05_Coords2Points %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS05_Coords2Points = function(coordslist,dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca y devuelve datos de los puntos solicitados en la base de datos.                    %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] coordslist: Lista de datos lon, lat (dataframe: double 3tr, double 3tr).             %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Lista de datos idGrid lat, lon, act (dataframe: integer, double 3tr,         %
  #              double 3tr, integer) (NULL: sin resultados).                                 %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste0("SELECT idGrid AS idGrid, latitud AS lat, longitud AS lon, activacion AS act FROM ",paste0(dbName,".",dbTable_Elevation),";")
  # Lanzar consulta:
  suppressWarnings({
	reallist = dbGetQuery(dbConnection, strQuery)
  })
  # Comparar listas:
  reallist$lat = trunc(reallist$lat*10^3)/10^3
  reallist$lon = trunc(reallist$lon*10^3)/10^3
  pointslist = semi_join(reallist,coordslist,by=c("lat","lon"))  
  # Terminar y devolver resultados:
  results = pointslist
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS05_ActivatePoints %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS05_ActivatePoints = function(nonactivelist,gridname,dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Activa puntos en la base de datos, insertando elevaciones, activacion y nombre de malla. %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] nonactivelist: Lista de datos idGrid (dataframe: integer).                    	      %
  #% [I] gridname: Nombre de malla de puntos (string).                                        %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Numero de puntos activados (integer) (NULL: sin resultados).                 %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta (solicitar puntos):
  strQuery = paste0("SELECT idGrid AS idGrid, latitud AS lat, longitud AS lon FROM ",paste0(dbName,".",dbTable_Elevation),";")
  # Lanzar consulta (solicitar puntos):
  suppressWarnings({
	reallist = dbGetQuery(dbConnection, strQuery)
  })
  # Solicitar datos de elevacion:
  nonactivelist = semi_join(reallist,nonactivelist,by="idGrid")
  nonactivecoords = data.frame(lat=nonactivelist$lat,lon=nonactivelist$lon,stringsAsFactors=FALSE)
  elevlist = GXS05_GetGoogleDEM(nonactivecoords)
  nonactivelist$lat = trunc(nonactivelist$lat*10^3)/10^3
  nonactivelist$lon = trunc(nonactivelist$lon*10^3)/10^3
  nonactlist = inner_join(nonactivelist,elevlist,by=c("lat","lon"))
  # Preparar cadena de valores (activar puntos):
  nonactlist$alt[nonactlist$alt<0] = 0
  active = 1
  vals = paste(toString(paste0("(",paste(nonactlist$idGrid,nonactlist$alt,paste0("'",gridname,"'"),paste0("'",active,"'"),sep=","),")")),sep=",")
  # Preparar cadena de consulta (activar puntos):
  strQuery = paste("INSERT INTO",paste0(dbName,".",dbTable_Elevation),"(idGrid,altitud,malla,activacion)",
                   "VALUES",vals,
                   "ON DUPLICATE KEY UPDATE altitud=VALUES(altitud),malla=VALUES(malla),activacion=VALUES(activacion);",collapse=" ")
  # Lanzar consulta (activar puntos):
  suppressWarnings({
    stmres = dbSendStatement(dbConnection, strQuery)
    dbClearResult(stmres)
  })
  # Terminar:
  results = length(nonactivelist$idGrid)
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS05_GetGoogleDEM %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS05_GetGoogleDEM = function(coords){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Solicita datos de elevacion a Google Maps Elevation API.                                 %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] coords: lista de datos lat, lon (dataframe: double 4, double 4).                     %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: lista de datos lat, lon, alt (dataframe: double 3tr, double 3tr, double 3r)  %
  #%             (NULL: sin resultados).                                                      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Configuracion de Google:
  APIURL = "https://maps.googleapis.com/maps/api/elevation/json?"
  APIKey = ""
  limpoints = 256 #512
  limsec = 100
  # Recortar dataframes de coordenadas:
  nchunks = ceiling(nrow(coords)/limpoints)
  coords = split(coords,rep(1:nchunks,each=limpoints)[1:nrow(coords)])
  # Solicitar datos de elevacion:
  results = data.frame(lat=character(),lon=character(),alt=character(),stringsAsFactors=FALSE)
  Sys.sleep(1)
  for (i in 1:nchunks){
    if (i%%limsec==0){Sys.sleep(1)}
    # Descargar datos en JSON:
    locs = paste(paste(coords[[i]][[1]],coords[[i]][[2]]),collapse="|")
    APIcall = paste0("'",APIURL,"locations=",locs,"&key=",APIKey,"'")
    wgetcall = paste0("wget -q -O - ",APIcall)
    # Guardar JSON en dataframe:
    jsonstr = paste(system(wgetcall,intern=TRUE,wait=TRUE),collapse="")
    df = fromJSON(jsonstr,simplify=FALSE,method="C")
    df = df$results
    res = data.frame(matrix(nrow=length(df),ncol=3),stringsAsFactors=FALSE)
    colnames(res) = c("lat","lon","alt")
    for (j in 1:length(df)){
      res[j,1] = formatC(df[[j]]$location$lat[1],digits=4,format="f")
      res[j,2] = formatC(df[[j]]$location$lng[1],digits=4,format="f")
      res[j,3] = formatC(df[[j]]$elevation[1],digits=4,format="f")
    }
    results = bind_rows(results,res)
  }
  # Dar formato a resultados:
  results$lat = trunc(as.numeric(results$lat)*10^3)/10^3
  results$lon = trunc(as.numeric(results$lon)*10^3)/10^3
  results$alt = round(as.numeric(results$alt),digits=3)  
  # Terminar y devolver resultados:
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
  } else if (key=="-gn"|key=="--gridname"){
    gridname = args[2];args = args[-1];args = args[-1]
  } else if (key=="-ct"|key=="--cortag"){
    cortag = args[2];args = args[-1];args = args[-1]
  } else if (key=="-it"|key=="--igntag"){
    igntag = args[2];args = args[-1];args = args[-1]
  } else if (key=="-lf"|key=="--logfile"){
    logfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-l"|key=="--lev"){
    lev = as.numeric(args[2])+1;args = args[-1];args = args[-1]
  } else {rSQL_ArgError()}
}
# Lanzar funcion principal:
if (exists("optfile") && exists("grib2file") && exists("pred") && exists("gridname") && exists("cortag") && exists("igntag")){
	GXS05_ToDataBase(grib2file,pred,gridname,cortag,igntag,scriptPath,optfile,logfile,lev)
  } else {rSQL_ArgError()}