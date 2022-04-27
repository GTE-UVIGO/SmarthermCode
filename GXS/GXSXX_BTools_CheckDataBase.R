#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_CheckDataBase %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_CheckDataBase = function(inidate="@begining",findate="@today",scriptPath,optfile,logfile="@console",lev=0){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca f-horas faltantes entre dos fechas limite.                                         %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] inidate: Fecha inicial del intervalo estudiado (string).                             %
  #% [I] findate: Fecha final del intervalo estudiado (string).                               %
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
  datafrec = GXSXX_ReadOptions("datafrec",optfile)
  days_lag = GXSXX_ReadOptions("days_lag",optfile)
  libpath = GXSXX_ReadOptions("libpath",optfile)
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("DBI")){require("DBI",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}			# Para trabajar con bases de datos
	if (GXSXX_CheckPack("RMySQL")){require("RMySQL",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para conectar con MySQL
  })
  # Mensajes de inicio de tarea:
  mText = paste0("GXSXX_BTools_CheckDataBase.")
  GXSXX_LogWrite(mText=mText,mType="T",mNewEx=TRUE,mLevel=lev,mIniFin="Ini",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  # Recopilar nombres de malla:
  gridlist = GXSXX_GetGrids(dbConnection,dbName,dbTable_Elevation)
  # Bucle de nombres de malla:
  for (gridname in gridlist$malla){
	# Procesar fechas solicitadas:
	if (inidate == "@begining"){inidate = GXSXX_GetIniDate(gridname,dbConnection,dbName,dbTable,dbTable_Elevation)}
	if (findate == "@today"){findate = Sys.Date()-as.numeric(days_lag)}
	# Buscar f-horas faltantes:
	missingfhours = GXSXX_CheckFHours(inidate,findate,gridname,dbConnection,dbName,dbTable,dbTable_Elevation,datafrec)
	cat(paste0("GXSXX_CheckDataBase: Huecos f-horarios para '",gridname,"'  entre ",inidate," y ",findate,".\n"))
	print(missingfhours)
  }
  # Terminar, cerrar conexion y devolver codigo de ejecucion correcta:
  dbDisconnect(dbConnection)
  mText = paste0("GXSXX_BTools_CheckDataBase.")
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Fin",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  return("[R]: 0")
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_GetGrids %
#%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_GetGrids = function(dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Devuelve los nombres de mallas definidas en la base de datos.                            %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Dataframe: malla (string) (NULL: sin resultados).                           %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste("SELECT DISTINCT malla FROM ",paste0(dbName,".",dbTable_Elevation)," WHERE malla <> '-';",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({
    d = dbGetQuery(dbConnection, strQuery)
  })
  results = data.frame("malla"=d,stringsAsFactors=FALSE)
  # Terminar y devolver resultados:
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_GetIniDate %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_GetIniDate = function(gridname,dbConnection,dbName,dbTable,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Devuelve la fecha inicial de registros en la base de datos para una malla concreta.      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] gridname: Nombre de malla estudiada (string).                                        %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable: Nombre de tabla de datos meteorologicos (string).             			  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Fecha de inicio de regristros (string) (NULL: sin resultados).              %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste("SELECT MIN(instante) AS inidate 
					FROM ",paste0(dbName,".",dbTable),"
					WHERE idGrid IN 
					  (SELECT idGrid 
					   FROM ",paste0(dbName,".",dbTable_Elevation),"
					   WHERE malla =",paste0("'",gridname,"'"),");",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({
    d = dbGetQuery(dbConnection, strQuery)
  })
  results = as.character(as.Date(d$inidate))  
  # Terminar y devolver resultados:
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_CheckFHours %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_CheckFHours = function(inidate,findate,gridname,dbConnection,dbName,dbTable,dbTable_Elevation,datafrec){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca f-horas faltantes entre dos fechas limite.                                         %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] inidate: Fecha inicial del intervalo estudiado (string).                             %
  #% [I] findate: Fecha final del intervalo estudiado (string).                               %
  #% [I] gridname: Nombre de malla estudiada (string).                                        %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable: Nombre de tabla de datos meteorologicos (string).             			  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #% [I] datafrec: Frecuencia de generacion de predicciones (integer).                        %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Dataframe: fecha (string), huecos (double) (NULL: sin resultados).          %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste("SELECT DATE(listahoras.fhoras) AS fecha, COUNT(listahoras.fhoras) AS huecos 
                    FROM 
                    	(SELECT datsec.horas AS fhoras 
                    	FROM 
                    		(SELECT SUBDATE(DATE_ADD('",findate,"', INTERVAL 23 HOUR), INTERVAL xc HOUR) AS horas 
                    		FROM 
                    			(SELECT @xi:=@xi+1 AS xc 
                    			FROM 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc1, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc2, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc3, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc4, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc5, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc6, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc7, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc8, 
                    				(SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 0) AS xc9, 
                    				(SELECT @xi:=-1) AS xc0 
                    			LIMIT ",as.numeric(as.Date(findate)-as.Date(inidate)+1)*24,") AS numsec) AS datsec 
                    	WHERE MOD(HOUR(datsec.horas),",datafrec,")=0 AND DATE(datsec.horas) BETWEEN ",paste0("'",inidate,"'")," AND ",paste0("'",inidate,"'"),") AS listahoras
                    LEFT JOIN 
						(SELECT fechas.fhoras1 AS fhoras 
						FROM 
							(SELECT idGrid, instante AS fhoras1 
							FROM ",paste0(dbName,".",dbTable),"
							WHERE MOD(HOUR(instante),",datafrec,")=0 AND DATE(instante) BETWEEN ",paste0("'",inidate,"'")," AND ",paste0("'",inidate,"'"),") AS fechas 
						JOIN 
						   (SELECT idGrid 
							FROM ",paste0(dbName,".",dbTable_Elevation),"
							WHERE malla = ",paste0("'",gridname,"'"),") AS puntos 
						ON fechas.idGrid = puntos.idGrid) AS datoshoras 
                    ON listahoras.fhoras = datoshoras.fhoras 
                    WHERE ISNULL(datoshoras.fhoras) 
                    GROUP BY DATE(listahoras.fhoras) 
                    ORDER BY listahoras.fhoras;",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({
    d = dbGetQuery(dbConnection, strQuery)
  })
  results = d
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
inidate = "@begining"
findate = "@today"
logfile = "@console"
lev = 0
# Procesar argumentos de funcion:
args = commandArgs(trailingOnly = TRUE)
while (length(args) > 0){
  key = args[1]
  if (key=="-of"|key=="--optfile"){
    optfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-id"|key=="--inidate"){
    inidate = args[2];args = args[-1];args = args[-1]
  } else if (key=="-fd"|key=="--findate"){
    findate = args[2];args = args[-1];args = args[-1]
  } else if (key=="-lf"|key=="--logfile"){
    logfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-l"|key=="--lev"){
    lev = as.numeric(args[2])+1;args = args[-1];args = args[-1]
  } else {rSQL_ArgError()}
}
# Lanzar funcion principal:
if (exists("optfile")){
	GXSXX_CheckDataBase(inidate,findate,scriptPath,optfile,logfile,lev)
  } else {rSQL_ArgError()}