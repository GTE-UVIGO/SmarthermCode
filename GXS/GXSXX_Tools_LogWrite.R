#!/usr/bin/env Rscript

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_LogWrite %
#%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_LogWrite = function(mText,mType="?",mLevel=0,mNewEx=FALSE,mIniFin="?",logfile="@default",scriptPath,optfile){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Escribe una linea de texto en un fichero.                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] mText: Texto del mensaje (string).                                                   %
  #% [I] mType: Etiqueta de tipo de entrada (string) (opcional).                              %
  #% [I] mLevel: Nivel de impresion de mensaje (integer) (opcional).                          %
  #% [I] mNewEx: Etiqueta de nueva ejecucion de codigo (boolean) (opcional).                  %
  #% [I] mIniFin: Etiqueta de inicio/final de proceso (string) (opcional).                    %
  #% [I] logfile: Nombre del fichero donde se escribe el mensaje (string) (opcional).         %
  #% [I] scriptPath: Ruta absoluta de ubicacion de scripts (string).                          %
  #% [I] optfile: Nombre del fichero de opciones globales (string).                           %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] GXSXX_RTools_CommonTools.R: Contiene funciones y variables comunes para ficheros R.  %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  mTelegram = FALSE
  levelmult = 2
  # Cargar scripts necesarios:
  source(paste0(scriptPath,"/GXSXX_RTools_CommonTools.R"))
  # Cargar variables globales:
  if (optfile=="@default"){
	logpath = scriptPath
	libpath = NULL
  } else {
	logpath = GXSXX_ReadOptions("logpath",optfile)
	libpath = GXSXX_ReadOptions("libpath",optfile)
  }
  # Cargar paquetes necesarios:
  suppressPackageStartupMessages({
	if (GXSXX_CheckPack("httr")){require("httr",lib.loc=libpath,quietly=TRUE,warn.conflicts=FALSE,character.only=TRUE)}	#Para poder usar wget y enviar alertas
  })
  # Marcar nueva ejecucion independiente:
  if (mNewEx){
    mNewEx = "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Nueva Ejecucion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
  } else {mNewEx = ""}
  # Preparar etiqueta de tipo de entrada:
  # [T] Task:     marca el fin de ejecucion de una tarea.
  # [I] Info:     marca un mensaje de informacion.
  # [O] Output:   marca una anotacion sobre resultados.
  # [W] Warning:  marca un aviso durante la ejecucion.
  # [E] Error:    marca un error durante la ejecucion.
  # [?] Other:    marca mensajes sin tipo o con un tipo incorrecto.
  if (mType=="T"|mType=="[T]"|mType=="Task"){
    mType = "[T]"
  } else if (mType=="I"|mType=="[I]"|mType=="Info") {
    mType = "[I]"
    mLevel = mLevel+1
  } else if (mType=="O"|mType=="[O]"|mType=="Output") {
    mType = "[O]"
    mLevel = mLevel+1
  } else if (mType=="W"|mType=="[W]"|mType=="Warning") {
    mType = "[W]"
    mLevel = mLevel+1
    mTelegram = TRUE
  } else if (mType=="E"|mType=="[E]"|mType=="Error") {
    mType = "[E]"
    mLevel = mLevel+1
    mTelegram = TRUE
  } else {
    mType = "[?]"
  }
  # Preparar etiqueta de instante:
  mtime = paste0("[",Sys.time(),"]")
  # Preparar texto:
  if (!substr(mText,nchar(mText),nchar(mText))=="."){mText=paste0(mText,".")}
  mText = paste0(paste0(rep(" ",times=mLevel*levelmult),collapse=""),mText)
  # Preparar etiqueta de inicio/final de entrada:
  # [Ini] Inicio: marca el inicio de una tarea o serie de tareas.
  # [Fin] Final:  marca el final de una tarea o serie de tareas.
  if (mIniFin=="Ini"|mIniFin=="[Ini]"|mIniFin=="Inicio"){
    mIniFin = "[Ini]"
  } else if (mIniFin=="Fin"|mIniFin=="[Fin]"|mIniFin=="Final") {
    mIniFin = "[Fin]"
  } else {
    mIniFin = "" 
  }
  # Preparar fichero de escritura:
  if (!substr(logpath,nchar(logpath),nchar(logpath))=="/"){logpath=paste0(logpath,"/")}
  if (optfile=="@default"){
    filelog = paste0(logpath,"logfile_default.log")
  } else if (logfile=="@dailylog"){
    filelog = paste0(logpath,"logfile_",format(Sys.Date(),"%Y_%m_%d"),".log")
  } else if (logfile=="@console"){
    filelog = ""
  } else {
    filelog = paste0(logpath,logfile)
  }
  if (!file.exists(filelog) && !logfile=="@console"){GXSXX_LogCreate(logfile=filelog)}
  # Actualizar contador de entradas:
  if (!logfile=="@console"){GXSXX_LogCounters(mType=mType,logfile=filelog)}
  # Imprimir mensaje:
  cat(mNewEx,mtime,mType,mText,mIniFin,"\n",file=filelog,sep=" ",fill=FALSE,append=TRUE)
  # Enviar aviso a Telegram:
  if (mTelegram){
    mTelebot = ""
    mTelechat = ""
    mTeleURL = paste0("https://api.telegram.org/bot",mTelebot,"/sendMessage?chat_id=",mTelechat,"&text=")
    mTelemess = paste(mType,trimws(mText))
    r = GET(paste0(mTeleURL,mTelemess))
    # Almacenar respuesta en caso de fallo:
    if (!r$status==200){
      # Preparar fichero de escritura:
      errfile = paste0(logpath,"errorfile.log")
      if (!file.exists(errfile)){file.create(errfile)}
      # Imprimir mensaje:
      cat(mNewEx,mtime,mType,mText,mIniFin,"| Status:",r$status,"\n",file=errfile,sep=" ",fill=FALSE,append=TRUE)
    }
  }
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_LogCreate %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_LogCreate = function(logfile){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Crea un fichero con un encabezado predeterminado.                                        %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] logfile: Ruta del fichero donde se escribe el mensaje (string) (opcional).           %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  file.create(logfile)
  Sys.chmod(paths=logfile,mode="777")
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%                                                                        %%   Archivo creado:   %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%                 Base GXS: Historial de acciones diarias                %%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%                                                                        %% ",format(Sys.time(),"%Y-%m-%d %H:%M:%S")," %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%% Contador: %% [T]:000000 %% [I]:000000 %% [O]:000000 %% [W]:000000 %% [E]:000000 %% [?]:000000 %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%% Etiquetas de tipo de entrada:                                                                 %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [T] Task:     marca el fin de ejecucion de una tarea.                                       %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [I] Info:     marca un mensaje de informacion.                                              %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [O] Output:   marca una anotacion sobre resultados.                                         %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [W] Warning:  marca un aviso durante la ejecucion.                                          %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [E] Error:    marca un error durante la ejecucion.                                          %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [?] Other:    marca mensajes sin tipo o con tipo incorrecto.                                %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%% Etiquetas de inicio/final de entrada:                                                         %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [Ini] Inicio: marca el inicio de una tarea o serie de tareas.                               %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [Fin] Final:  marca el final de una tarea o serie de tareas.                                %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%% Notas y Comentarios:                                                                          %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%   [*] Un error con esta marca dentro de un bucle no interrumpe dicho bucle.                   %%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n",file=logfile,sep="",fill=FALSE,append=TRUE)
  cat("\n",file=logfile,sep="",fill=FALSE,append=TRUE)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Funcion GXSXX_LogCounters %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GXSXX_LogCounters = function(mType,logfile){
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% Actualiza los contadores de entradas en un fichero.                                      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [I] mType: Etiqueta de tipo de entrada (string).                                         %
  #% [I] logfile: Ruta del fichero donde se escribe el mensaje (string).                      %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [S] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  #% [O] None.                                                                                %
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Leer fichero:
  filelines = readLines(logfile,-1)
  # Editar fila de contadores:
  regex = paste0("\\",mType,":......")
  countersline = filelines[7]
  countero = regmatches(countersline,regexpr(regex,countersline))
  countn = formatC(as.numeric(substr(countero,5,10))+1,width=nchar(substr(countero,5,10)),format="d",flag="0")
  countern = paste0(mType,":",countn)
  countersline = gsub(regex,countern,countersline)
  filelines[7] = countersline
  # Reescribir fichero:
  writeLines(filelines,logfile)
}
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$ Lanzar funcion desde Bash $
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Filtrar sources desde otros archivos .R:
callingfile = basename(commandArgs(trailingOnly = FALSE)[4])
if (callingfile=="GXSXX_Tools_LogWrite.R"){
  # Determinar directorio de ejecucion:
  match = grep("--file=",commandArgs(trailingOnly=FALSE))
  scriptPath = dirname(normalizePath(sub("--file=","",commandArgs(trailingOnly=FALSE)[match])))
  # Valores por defecto:
  mType = "?"
  mLevel = 0
  mNewEx = FALSE
  mIniFin = "?"
  logfile = "@default"
  # Procesar argumentos de funcion:
  args = commandArgs(trailingOnly = TRUE)
  while (length(args) > 0){
    key = args[1]
    if (key=="-m"|key=="--mText"){
      mText = args[2];args = args[-1];args = args[-1]
    } else if (key=="-mt"|key=="--mType"){
      mType = args[2];args = args[-1];args = args[-1]
    } else if (key=="-ml"|key=="--mLevel"){
      mLevel = as.numeric(args[2]);args = args[-1];args = args[-1]
    } else if (key=="-mn"|key=="--mNewEx"){
      mNewEx = as.logical(args[2]);args = args[-1];args = args[-1]
    } else if (key=="-mi"|key=="--mIniFin"){
      mIniFin = args[2];args = args[-1];args = args[-1]
    } else if (key=="-lf"|key=="--logfile"){
      logfile = args[2];args = args[-1];args = args[-1]
    } else if (key=="-of"|key=="--optfile"){
      optfile = args[2];args = args[-1];args = args[-1]
    } else {
      GXSXX_LogWrite(paste0("GXSTT ERROR: Se ha proporcionado una flag '",args[1],"' no reconocida. Operacion cancelada."),mType="E",logfile=logfile,optfile="@default")
      stop()
    }
  }
  # Lanzar funcion principal:
  if (exists("mText") && exists("optfile")){
	GXSXX_LogWrite(mText,mType,mLevel,mNewEx,mIniFin,logfile,scriptPath,optfile)
  } else {
	GXSXX_LogWrite(paste0("GXSTT ERROR: No se han proporcionado suficientes datos para lanzar GXSXX_LogWrite."),mType="E",logfile=logfile,optfile="@default")
	stop()
  }
}