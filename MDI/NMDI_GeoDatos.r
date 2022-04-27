#!/usr/bin/env Rscript
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_mallaubicaciones %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_mallaubicaciones = function(latmin,latmax,lonmin,lonmax,resolucion,
	proyeccion="+init=epsg:4326",reproyeccion=proyeccion,ids=NULL,
	malla=""){
	# Cargar scripts externos:
	source("./MDIXX_GlobalData.R")
	# Cargar variables globales:
	meshpath = MDIXX_GlobalData$Directorios$Ruta_Mallas
	# Buscar malla preexistente:
	if (!malla==""){
		dir.create(meshpath,showWarnings=FALSE)
		if (!substr(malla,nchar(malla)-nchar(".rds")+1,
			nchar(malla))==".rds"){malla=paste0(malla,".rds")}
		meshfile = file.path(meshpath,malla)
		if (file.exists(meshfile)){
			result = readRDS(meshfile)
			# Terminar y devolver resultados:
			return(result)}}
	# Generar malla:
	ubicaciones = MDI_geomalla(latmin,latmax,lonmin,lonmax,resolucion,
		proyeccion,reproyeccion)
	ubicaciones = MDI_geoatributos(ubicaciones,ids)
	# Guardar malla:
	if (!malla==""){
		saveRDS(ubicaciones,file=meshfile)}
	# Terminar y devolver resultados:
	return(ubicaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_listaubicaciones %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_listaubicaciones = function(latitudes,longitudes,
	proyeccion="+init=epsg:4326",reproyeccion=proyeccion,ids=NULL,
	malla=""){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	meshpath = MDIXX_GlobalData$Directorios$Ruta_Mallas
	# Buscar malla preexistente:
	if (!malla==""){
		dir.create(meshpath,showWarnings=FALSE)
		if (!substr(malla,nchar(malla)-nchar(".rds")+1,
			nchar(malla))==".rds"){malla=paste0(malla,".rds")}
		meshfile = file.path(meshpath,malla)
		if (file.exists(meshfile)){
			result = readRDS(meshfile)
			# Terminar y devolver resultados:
			return(result)}}
	# Generar malla:
	ubicaciones = MDI_geopuntos(latitudes,longitudes,proyeccion,
		reproyeccion)
	ubicaciones = MDI_geoatributos(ubicaciones,ids)
	# Guardar malla:
	if (!malla==""){
		saveRDS(ubicaciones,file=meshfile)}
	# Terminar y devolver resultados:
	return(ubicaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_mallaobservaciones %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_mallaobservaciones = function(ubicaciones,fuente,varsolic,inicio,
	final=inicio,radio="",disponibilidad="",omision="",ids=NULL,
	malla=""){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	meshpath = MDIXX_GlobalData$Directorios$Ruta_Mallas
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp",lib.loc=libpath)})
	# Buscar malla preexistente:
	if (!malla==""){
		dir.create(meshpath,showWarnings=FALSE)
		if (!substr(malla,nchar(malla)-nchar(".rds")+1,
			nchar(malla))==".rds"){malla=paste0(malla,".rds")}
		meshfile = file.path(meshpath,malla)
		if (file.exists(meshfile)){
			result = readRDS(meshfile)
			# Terminar y devolver resultados:
			return(result)}}
	# Generar malla:
	puntos = MDI_puntosobservacion(ubicaciones,fuente,varsolic,inicio,final,
		radio,disponibilidad,omision)
	observaciones = MDI_geopuntos(latitudes=puntos$Latitud,
		longitudes=puntos$Longitud,reproyeccion=proj4string(ubicaciones))
	observaciones = sp::merge(observaciones,puntos,by=c("Longitud","Latitud"))
	observaciones = MDI_geoatributos(observaciones,ids)
	# Guardar malla:
	if (!malla==""){
		saveRDS(observaciones,file=meshfile)}
	# Terminar y devolver resultados:
	return(observaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_puntosobservacion %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_puntosobservacion = function(ubicaciones,fuente,varsolic,inicio,
	final=inicio,radio="",disponibilidad="",omision=""){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	if (fuente=="MG"){
		tabladatos = MDIXX_GlobalData$MySQL_Tables$MG_Data
		tablapuntos = MDIXX_GlobalData$MySQL_Tables$MG_Grid
		dbConnection = MDIXX_GlobalData$MySQL_Conn$MG
	} else if (fuente=="GFS"){
		tabladatos = MDIXX_GlobalData$MySQL_Tables$GFS_Data
		tablapuntos = MDIXX_GlobalData$MySQL_Tables$GFS_Grid
		dbConnection = MDIXX_GlobalData$MySQL_Conn$GXS
	} else {stop(paste0("MDI01 ERROR: No se reconoce la fuente de datos '",
		fuente,"'."))}
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp",lib.loc=libpath)})
	# Preparar consulta para nombres de campos:
	if (fuente=="MG"){campoid = "estID"
	} else if (fuente=="GFS"){campoid = "idGrid"}
	# Preparar consulta para variables solicitadas:
	if (toupper(fuente)=="GFS"){varsolic=c("HGT",varsolic)}
	strSELECTsoli = paste0(", ",paste0("SUM(!ISNULL(meteo.",varsolic,
		")) AS ",varsolic,collapse=", ")," ")
	# Preparar consulta para límites temporales:
	minins = as.character(inicio)
	maxins = as.character(final)
	strWHEREinst = ifelse(maxins==minins,paste0("(meteo.instante = '",
		maxins,"') "),paste0("(meteo.instante BETWEEN '",minins,"' AND '",
		maxins,"') "))
	# Preparar consulta para filtro de radio:
	if (!radio==""){
		radio = as.numeric(radio)
		minlat = min(ubicaciones$Latitud) - radio
		maxlat = max(ubicaciones$Latitud) + radio
		minlon = min(ubicaciones$Longitud) - radio
		maxlon = max(ubicaciones$Longitud) + radio
		strWHEREarea = paste0("AND (puntos.longitud <= ",maxlon,
			" AND puntos.longitud >= ",minlon," AND puntos.latitud <= ",
			maxlat," AND puntos.latitud >= ",minlat,") ")
	} else {strWHEREarea = NULL}
	# Preparar consulta para filtro de omisiones:
	omitidos = paste0("(",paste0(unlist(omision),collapse=","),")")
	strWHEREomit = ifelse((omision==""),"",paste0(" AND puntos.",campoid,
		" NOT IN ",omitidos))
	# Lanzar consulta:
	strQuery = paste0(
		"SELECT meteo.",campoid," AS Codigo",strSELECTsoli,", ",
		"puntos.",campoid," AS Codigo, puntos.latitud AS Latitud, 
			puntos.longitud AS Longitud ",
		"FROM ",tabladatos$bdd,".",tabladatos$tabla," AS meteo ",
		"JOIN ",tablapuntos$bdd,".",tablapuntos$tabla," AS puntos ",
		"ON (meteo.",campoid," = puntos.",campoid,") ",
		"WHERE ",strWHEREinst,strWHEREarea,strWHEREomit,
		"GROUP BY meteo.",campoid,
		";")
	puntos = MDIXX_selectMySQL(dbConnection,strQuery)
	# Filtrar estaciones según disponibilidad:
	if (!disponibilidad==""){
		disponibilidad = as.numeric(disponibilidad)
		intervalo = as.integer(difftime(final,inicio,units="hours")+1)
		puntos[,varsolic] = puntos[,varsolic]/intervalo*100
		puntos = puntos[apply(puntos[,varsolic] >= disponibilidad,
			1,all),c("Codigo","Latitud","Longitud")]}
	else {puntos = puntos[,c("Codigo","Latitud","Longitud")]}
	# Terminar y devolver resultados:
	return(puntos)
}
#%%%%%%%%%%%%%%%%
#% MDI_geomalla %
#%%%%%%%%%%%%%%%%
MDI_geomalla = function(latmin,latmax,lonmin,lonmax,resolucion,
	proyeccion="+init=epsg:4326",reproyeccion=proyeccion){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp",lib.loc=libpath)
		require("rgdal",lib.loc=libpath)
		require("raster",lib.loc=libpath)
	})
	# Definir extremos de malla:
	geopuntos = data.frame(Latitud=c(latmin,latmax),
						Longitud=c(lonmin,lonmax))
	sp::coordinates(geopuntos) = data.frame(x=geopuntos$Longitud,
										y=geopuntos$Latitud)
	sp::proj4string(geopuntos) = sp::CRS(proyeccion)
	if (!reproyeccion==proyeccion){
		geopuntos = sp::spTransform(geopuntos,CRS=reproyeccion)}
	# Definir malla:
	geomalla = expand.grid(
		x=seq(raster::xmin(geopuntos)+resolucion/2,
			  raster::xmax(geopuntos)-resolucion/10,by=resolucion),
		y=seq(raster::ymin(geopuntos)+resolucion/2,
			  raster::ymax(geopuntos)-resolucion/10,by=resolucion))
	sp::coordinates(geomalla) = ~x + y
	sp::proj4string(geomalla) = sp::proj4string(geopuntos)
	sp::gridded(geomalla) = TRUE
	# Calcular coordenadas geográficas de puntos:
	lonlat = data.frame(geomalla@coords)
	sp::coordinates(lonlat) = data.frame(geomalla@coords)
	sp::proj4string(lonlat) = sp::CRS(reproyeccion)
	if (!reproyeccion==proyeccion){
		lonlat = sp::spTransform(lonlat,CRS=proyeccion)}
	geomalla$Latitud = lonlat$y
	geomalla$Longitud = lonlat$x
	# Terminar y devolver resultados:
	return(geomalla)
}
#%%%%%%%%%%%%%%%%%
#% MDI_geopuntos %
#%%%%%%%%%%%%%%%%%
MDI_geopuntos = function(latitudes,longitudes,proyeccion="+init=epsg:4326",
	reproyeccion=proyeccion){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp",lib.loc=libpath)
		require("rgdal",lib.loc=libpath)
	})
	# Definir puntos:
	geopuntos = data.frame(Latitud=latitudes,Longitud=longitudes)
	sp::coordinates(geopuntos) = data.frame(x=geopuntos$Longitud,
										y=geopuntos$Latitud)
	sp::proj4string(geopuntos) = sp::CRS(proyeccion)
	if (!reproyeccion==proyeccion){
		geopuntos = sp::spTransform(geopuntos,CRS=reproyeccion)}
	# Terminar y devolver resultados:
	return(geopuntos)
}
#%%%%%%%%%%%%%%%%%%%%
#% MDI_geoatributos %
#%%%%%%%%%%%%%%%%%%%%
MDI_geoatributos = function(geopuntos,ids=NULL,sinmar=FALSE){
	# Cargar scripts externos:
	source("")
	# Eliminar ubicaciones en mar:
	if (sinmar){
		if (is.null(geopuntos$Tierra) || anyNA(geopuntos$Tierra)){
			valores = tierra(geopuntos@coords,proj4string(geopuntos))
			if (anyNA(valores)){warning(paste0("AVISO: No se ha podido 
				asignar categoría de tierra o mar a ",sum(is.na(valores)),
				" coordenadas."))}
			geopuntos$Tierra = valores}
		geopuntos = geopuntos[geopuntos$Tierra==TRUE,]
		geopuntos$Tierra = NULL}
	# Generar columna de identificación:
	if (!is.null(ids)){
		geopuntos$Id = NULL
		geopuntos$Codigo = NULL
		if (!length(ids)==nrow(geopuntos)){
			stop("ERROR: La longitud del vector 'ids' no coincide con
				el número de filas del dataframe 'geopuntos' (",length(ids),
				" frente a ",nrow(geopuntos),"). Ejecución cancelada.")}
		geopuntos$Id = ids}
	else if (is.null(geopuntos$Id) & is.null(geopuntos$Codigo)){
		geopuntos$Id = seq(nrow(geopuntos))}
	# Calcular altitudes desconocidas:
	if (is.null(geopuntos$Altitud) || anyNA(geopuntos$Altitud)){
		valores = altitudes(geopuntos@coords,proj4string(geopuntos))
		if (anyNA(valores)){warning(paste0("AVISO: No se ha podido asignar 
			elevación a ",sum(is.na(valores))," coordenadas."))}
	geopuntos$Altitud = valores}
	# Calcular distancias a costa desconocidas:
	if (is.null(geopuntos$distCosta) || anyNA(geopuntos$distCosta)){
		valores = distanciascosta(geopuntos@coords,proj4string(geopuntos))
		if (anyNA(valores)){warning(paste0("AVISO: No se ha podido asignar 
			distancia a costa a ",sum(is.na(valores))," coordenadas."))}
		geopuntos$distCosta = valores}
	# Reordenar columnas de atributos:
	ncs = unique(c(match(c("Id","Codigo","Latitud","Longitud","Altitud",
		"distCosta"),colnames(geopuntos@data)),1:ncol(geopuntos)))
	geopuntos = geopuntos[,ncs[!is.na(ncs)]]
	# Terminar y devolver resultados:
	return(geopuntos)
}