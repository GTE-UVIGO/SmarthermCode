#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS07_GetList %
#%%%%%%%%%%%%%%%%%%%%%%%%%
GXS07_GetList = function(date,ndays,cortag,igntag,scriptPath,optfile,logfile="@console",lev=0){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca errores de datos dentro de un intervalo de X dias.                                 %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] date: Fecha en la cual termina el intervalo de X dias considerado (string).          %
  #% [I] ndays: Tamanho del intervalo de X dias considerado (string).                         %
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
  #% [O] results: Vector de dias con f-horas incorrectas (string) (NULL: sin resultados).     %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Cargar scripts necesarios:
  source(paste0(scriptPath,"/GXSXX_Tools_LogWrite.R"))
  source(paste0(scriptPath,"/GXSXX_RTools_CommonTools.R"))
  # Cargar variables globales:
  datafrec = GXSXX_ReadOptions("datafrec",optfile)
  dbConnection = GXSXX_OpenConnection(optfile)
  dbName = GXSXX_ReadOptions("dbName",optfile)
  dbTable = GXSXX_ReadOptions("dbTable",optfile)
  dbTable_Elevation = GXSXX_ReadOptions("dbTable_Elevation",optfile)
  libpath = GXSXX_ReadOptions("libpath",optfile)
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("DBI")){require("DBI",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para trabajar con bases de datos
	if (GXSXX_CheckPack("RMySQL")){require("RMySQL",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}	# Para conectar con MySQL
  })
  # Mensajes de inicio de tarea:
  mText = "GXS07_Gaps_GetErrorDays."
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Ini",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Buscar f-horas faltantes %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  d1 = GXS07_MissingHours(date,ndays,dbConnection,dbName,dbTable,datafrec)
  c1 = formatC(nrow(d1),width=nchar(ndays),format="d",flag="0")
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Buscar NULLs en valores meteo %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  d2 = GXS07_NullValues(date,ndays,dbConnection,dbName,dbTable,datafrec)
  c2 = formatC(nrow(d2),width=nchar(ndays),format="d",flag="0")
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Buscar tags de error en valores meteo %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  d3 = GXS07_ErrorTags(date,ndays,cortag,igntag,dbConnection,dbName,dbTable,datafrec)
  c3 = formatC(nrow(d3),width=nchar(ndays),format="d",flag="0")
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Generar lista combinada sin duplicados %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  # Resumen de dias con errores:
  mText = paste0("Dias con errores: ",c1," (horas) | ",c2," (NULLs) | ",c3," (tags).")
  GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)  
  # Generar lista:
  df = merge(d1,d2,all=TRUE)
  df = merge(df,d3,all=TRUE)
  dfs = toString(df$errordays)
  results = paste0("[O] ",gsub(",","",dfs)," [O] ","[R]: 0 [R]")
  # Terminar y devolver resultados:
  mText = "GXS07_Gaps_GetErrorDays."
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Fin",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  if (is.null(results)) return("[R]: 0 [R]") else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS07_MissingHours %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS07_MissingHours = function(date,ndays,dbConnection,dbName,dbTable,datafrec){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca f-horas faltantes dentro de un intervalo de X dias.                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] date: Fecha en la cual termina el intervalo de X dias considerado (string).          %
  #% [I] ndays: Tamanho del intervalo de X dias considerado (string).                         %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable: Nombre de tabla de datos meteoroligicos (string).                       	  %
  #% [I] datafrec: Frecuencia de predicciones en horas (integer).							  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Lista de dias con f-horas incorrectas (dataframe) (NULL: sin resultados).   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste("SELECT DATE(listahoras.fhoras) AS errordays
                    FROM
                        (SELECT horas AS fhoras
                        FROM
                            (SELECT SUBDATE(DATE_ADD('",date,"', INTERVAL 23 HOUR), INTERVAL xc HOUR) AS horas
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
                                LIMIT ",24*as.numeric(ndays),") sec) lis
                        WHERE MOD(HOUR(horas),",datafrec,")=0)
                        AS listahoras
                    LEFT JOIN
                        (SELECT instante AS fhoras
                        FROM ",dbName,".",dbTable,"
                        WHERE MOD(HOUR(instante),",datafrec,")=0 AND DATE(instante) BETWEEN SUBDATE('",date,"', INTERVAL ",ndays," DAY) AND DATE('",date,"')
                        GROUP BY instante) AS datoshoras
                    ON listahoras.fhoras = datoshoras.fhoras
                    WHERE ISNULL(datoshoras.fhoras)
                    GROUP BY DATE(listahoras.fhoras)
                    ORDER BY DATE(listahoras.fhoras);",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({d = dbGetQuery(dbConnection, strQuery)})
  results = d
  # Terminar y devolver resultados:
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS07_NullValues %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS07_NullValues = function(date,ndays,dbConnection,dbName,dbTable,datafrec){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca f-horas con valores nulos en variables meteo dentro de un intervalo de X dias.     %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] date: Fecha en la cual termina el intervalo de X dias considerado (string).          %
  #% [I] ndays: Tamanho del intervalo de X dias considerado (string).                         %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable: Nombre de tabla de datos meteoroligicos (string).                       	  %
  #% [I] datafrec: Frecuencia de predicciones en horas (integer).							  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Lista de dias con f-horas incorrectas (dataframe) (NULL: sin resultados).   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  strQuery = paste("SELECT DATE(instante) AS errordays
                   FROM ",dbName,".",dbTable," 
                   WHERE MOD(HOUR(instante),",datafrec,") = 0 AND DATE(instante) BETWEEN SUBDATE('",date,"', INTERVAL ",ndays," DAY) AND DATE('",date,"')
                   AND (ISNULL(TA) OR ISNULL(HR) OR ISNULL(PR) OR ISNULL(RS) OR ISNULL(VV) OR ISNULL(DV) OR ISNULL(PP))   
                   GROUP BY DATE(instante)
                   ORDER BY DATE(instante);",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({d = dbGetQuery(dbConnection, strQuery)})
  results = d
  # Terminar y devolver resultados:
  if (is.null(results)) return(NULL) else return(results)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXS07_ErrorTags %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXS07_ErrorTags = function(date,ndays,cortag,igntag,dbConnection,dbName,dbTable,datafrec){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca f-horas con tags de error para variables meteo dentro de un intervalo de X dias.   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] date: Fecha en la cual termina el intervalo de X dias considerado (string).          %
  #% [I] ndays: Tamanho del intervalo de X dias considerado (string).                         %
  #% [I] cortag: Valor del tag para datos correctos (integer).                                %
  #% [I] igntag: Valor del tag para datos ignorados (integer).                                %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable: Nombre de tabla de datos meteoroligicos (string).                       	  %
  #% [I] datafrec: Frecuencia de predicciones en horas (integer).							  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] results: Lista de dias con f-horas incorrectas (dataframe) (NULL: sin resultados).   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar cadena de consulta:
  tags = c("TA_tg","HR_tg","PR_tg","RS_tg","VV_tg","DV_tg","PP_tg")
  strTags = paste(lapply(tags, function(tag){sprintf("(%s != %s AND %s != %s)",
	tag,cortag,tag,igntag)}),collapse = " OR ")
  strQuery = paste("SELECT DATE(instante) AS errordays
                    FROM ",dbName,".",dbTable," 
                    WHERE MOD(HOUR(instante),",datafrec,") = 0 AND DATE(instante) BETWEEN SUBDATE('",date,"', INTERVAL ",ndays," DAY) AND DATE('",date,"')
                    AND (",strTags,")
                    GROUP BY DATE(instante)
                    ORDER BY DATE(instante);",collapse=" ")
  # Lanzar consulta:
  suppressWarnings({d = dbGetQuery(dbConnection, strQuery)})
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
logfile = "@console"
lev = 0
# Procesar argumentos:
args = commandArgs(trailingOnly = TRUE)
while (length(args) > 0){
  key = args[1]
  if (key=="-of"|key=="--optfile"){
    optfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-d"|key=="--date"){
    date = args[2];args = args[-1];args = args[-1]
  } else if (key=="-n"|key=="--ndays"){
    ndays = args[2];args = args[-1];args = args[-1]
  } else if (key=="-ct"|key=="--cortag"){
    cortag = args[2];args = args[-1];args = args[-1]
  } else if (key=="-it"|key=="--igntag"){
    igntag = args[2];args = args[-1];args = args[-1]
  } else if (key=="-lf"|key=="--logfile"){
    logfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-l"|key=="--lev"){
    lev = as.numeric(args[2])+1;args = args[-1];args = args[-1]
  } else {
    rSQL_ArgError()
  }
}
# Lanzar funcion principal:
if (exists("optfile") && exists("date") && exists("ndays") && exists("cortag") && exists("igntag")){
	GXS07_GetList(date,ndays,cortag,igntag,scriptPath,optfile,logfile,lev)
  } else {rSQL_ArgError()}