#!/usr/bin/env Rscript
#%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_geotiff %
#%%%%%%%%%%%%%%%%%%%%%%%
MDI_geotiff = function(estimaciones, nombre){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp",lib.loc=libpath)
		require("raster",lib.loc=libpath)
	})
	# Identificar variables:
	identificadores = c("Id","Codigo")
	instante = c("Instante")
	predictores = c("Latitud","Longitud","Altitud","distCosta")
	variables = names(estimaciones)[!names(estimaciones) %in% 
		c(identificadores,instante,predictores)]
	# Transformar a pila raster:
	ras = lapply(variables,function(var){raster(estimaciones[,var])})
	sta = stack(ras)
	# Generar archivo GeoTIFF:
	nombre = paste0(nombre,".tif")
	writeRaster(sta,filename=nombre,format="GTiff",
				overwrite=TRUE,bylayer=FALSE,silent=TRUE)
 }