#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_ManageGrid %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_ManageGrid = function(gridname,action,scriptPath,optfile,logfile="@console",lev=0){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Administra la activacion ad-hoc de mallas preexistentes en la base de datos.			  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] gridname: Nombre de malla de puntos (string).                                        %
  #% [I] action: Accion a ejecutar (string).												  %
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
  dbTable_Elevation = GXSXX_ReadOptions("dbTable_Elevation",optfile)
  libpath = GXSXX_ReadOptions("libpath",optfile) 
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("DBI")){require("DBI",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}			# Para trabajar con bases de datos
	if (GXSXX_CheckPack("RMySQL")){require("RMySQL",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para conectar con MySQL
  })
  # Mensajes de inicio de tarea:
  mText = paste0("GXSXX_BTools_ManageGrid.")
  GXSXX_LogWrite(mText=mText,mType="T",mNewEx=TRUE,mLevel=lev,mIniFin="Ini",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Procesar accion solicitada %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if (action=="activar") 			{res = GXSXX_ActivateGrid(gridname,activate=TRUE,dbConnection,dbName,dbTable_Elevation)
  } else if (action=="desactivar")  {res = GXSXX_ActivateGrid(gridname,activate=FALSE,dbConnection,dbName,dbTable_Elevation)
  } else {
	mText = paste0("GXSXX ERROR: No se reconoce la accion '",action,"' solicitada.")
	GXSXX_LogWrite(mText=mText,mType="E",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)
	return("[R]: 101")
  }
  # Imprimir resultados:
  mText = paste0("Numero de filas afectadas por la accion '",action,"': ",res,".")
  GXSXX_LogWrite(mText=mText,mType="I",mLevel=lev,logfile=logfile,scriptPath=scriptPath,optfile=optfile)	
  # Terminar, cerrar conexion y devolver codigo de ejecucion correcta:
  dbDisconnect(dbConnection)
  mText = paste0("GXSXX_BTools_ManageGrid.")
  GXSXX_LogWrite(mText=mText,mType="T",mLevel=lev,mIniFin="Fin",logfile=logfile,scriptPath=scriptPath,optfile=optfile)
  return("[R]: 0")
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_ActivateGrid %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_ActivateGrid = function(gridname,activate=TRUE,dbConnection,dbName,dbTable_Elevation){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Busca puntos no existentes en la base de datos.                                          %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] gridname: Nombre de malla de puntos (string).                                        %
  #% [I] activate: Activar o desactivar malla de puntos (booleano) (opcional).                %
  #% [I] dbConnection: Objeto conexion con base de datos (MySQLConnection).                   %
  #% [I] dbName: Nombre de base de datos (string).											  %
  #% [I] dbTable_Elevation: Nombre de tabla de datos de malla (string).                       %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: numero de puntos activados o desactivados (integer) (NULL: sin resultados).  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Preparar valores:
  actnum = ifelse(activate,1,0)
  # Preparar cadenas de consultas:
  strQuery = paste0("UPDATE ",dbName,".",dbTable_Elevation," SET activacion = '",actnum,"' WHERE malla = '",gridname,"';")
  # Lanzar consultas:
  suppressWarnings({
    stmres = dbSendStatement(dbConnection, strQuery)
	affrows = dbGetRowsAffected(stmres)
    dbClearResult(stmres)
  })
  # Terminar y devolver resultados:
  results = affrows
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
  } else if (key=="-gn"|key=="--gridname"){
    gridname = args[2];args = args[-1];args = args[-1]
  } else if (key=="-ac"|key=="--action"){
    action = args[2];args = args[-1];args = args[-1]
  } else if (key=="-lf"|key=="--logfile"){
    logfile = args[2];args = args[-1];args = args[-1]
  } else if (key=="-l"|key=="--lev"){
    lev = as.numeric(args[2])+1;args = args[-1];args = args[-1]
  } else {rSQL_ArgError()}
}
# Lanzar funcion principal:
if (exists("optfile") && exists("gridname") && exists("action")){
	GXSXX_ManageGrid(gridname,action,scriptPath,optfile,logfile,lev)
  } else {rSQL_ArgError()}