#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_ReadOptions %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_ReadOptions = function(variable,optfile){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Lee un archivo txt que contiene variables comunes para varios scripts.                   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] variable: Nombre de la variable solicitada (string).                                 %
  #% [I] optfile: Nombre del fichero txt procesado (string).                                  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Valor de la variable solicitada (string) (NULL: sin resultados).             %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Leer archivo de opciones:
  filelines = readLines(optfile,-1)
  for (i in 1:length(filelines)){
    if (substr(filelines[i],1,nchar("dbHostName="))=="dbHostName=")                        	{dbHostName = gsub("\"","",substr(filelines[i],nchar("dbHostName=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("dbPortNumber="))=="dbPortNumber=")             	{dbPortNumber = as.numeric(gsub("\"","",substr(filelines[i],nchar("dbPortNumber=")+1,nchar(filelines[i]))))
    } else if (substr(filelines[i],1,nchar("dbUserName="))=="dbUserName=")                 	{dbUserName = gsub("\"","",substr(filelines[i],nchar("dbUserName=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("dbPassword="))=="dbPassword=")                 	{dbPassword = gsub("\"","",substr(filelines[i],nchar("dbPassword=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("dbName="))=="dbName=")                         	{dbName = gsub("\"","",substr(filelines[i],nchar("dbName=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("dbTable="))=="dbTable=")                   		{dbTable = gsub("\"","",substr(filelines[i],nchar("dbTable=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("dbTable_Elevation="))=="dbTable_Elevation=") 	{dbTable_Elevation = gsub("\"","",substr(filelines[i],nchar("dbTable_Elevation=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("libpath="))=="libpath=")                       	{libpath = gsub("\"","",substr(filelines[i],nchar("libpath=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("logpath="))=="logpath=")                       	{logpath = gsub("\"","",substr(filelines[i],nchar("logpath=")+1,nchar(filelines[i])))
	} else if (substr(filelines[i],1,nchar("datafrec="))=="datafrec=")                   	{datafrec = gsub("\"","",substr(filelines[i],nchar("datafrec=")+1,nchar(filelines[i])))
    } else if (substr(filelines[i],1,nchar("days_lag="))=="days_lag=")                   	{days_lag = gsub("\"","",substr(filelines[i],nchar("days_lag=")+1,nchar(filelines[i])))
	}
  }
  # Seleccionar variable solicitada:
  if (variable=="dbHostName") {result=dbHostName
  } else if (variable=="dbPortNumber") {result=dbPortNumber
  } else if (variable=="dbUserName") {result=dbUserName
  } else if (variable=="dbPassword") {result=dbPassword
  } else if (variable=="dbName") {result=dbName
  } else if (variable=="dbTable") {result=dbTable
  } else if (variable=="dbTable_Elevation") {result=dbTable_Elevation
  } else if (variable=="libpath") {result=libpath
  } else if (variable=="logpath") {result=logpath
  } else if (variable=="datafrec") {result=datafrec
  } else if (variable=="days_lag") {result=days_lag
  } else {stop()}
  # Devolver resultado:
  return(result)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_OpenConnection %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_OpenConnection = function(optfile){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Abre una conexion con una base de datos usando las variables de un fichero de opciones.  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] optfile: Nombre del fichero txt procesado (string).                                  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] return: Objeto conexion con base de datos (MySQLConnection) (NULL: sin resultados).  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Solicitar datos de conexion:
  dbHostName = GXSXX_ReadOptions("dbHostName",optfile)
  dbPortNumber = GXSXX_ReadOptions("dbPortNumber",optfile)
  dbUserName = GXSXX_ReadOptions("dbUserName",optfile)
  dbPassword = GXSXX_ReadOptions("dbPassword",optfile)
  dbName = GXSXX_ReadOptions("dbName",optfile)
  libpath = GXSXX_ReadOptions("libpath",optfile)
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("DBI")){require("DBI",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}		# Para trabajar con bases de datos
	if (GXSXX_CheckPack("RMySQL")){require("RMySQL",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}	# Para conectar con MySQL
	})
  # Abrir conexion:
  dbConnection = dbConnect(MySQL(),user=dbUserName,password=dbPassword,dbname=dbName,host=dbHostName,port=dbPortNumber)
  # Devolver conexion:
  return(dbConnection)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_CheckPack %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_CheckPack = function(package){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Comprueba si un paquete de funciones R puede ser cargado.								  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] package: Nombre del paquete R solicitado (string).                                   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] result: Posibilidad de carga de paquete (boolean).                                   %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  result = ifelse(length(grep(package,loadedNamespaces(),fixed=TRUE))>0,FALSE,TRUE)
  return(result)
}