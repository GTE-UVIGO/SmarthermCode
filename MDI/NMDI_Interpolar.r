#!/usr/bin/env Rscript
#%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_interpolar %
#%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_interpolar = function(observaciones, ubicaciones, metodo){
	# Seleccionar función interpoladora:
	if (metodo %in% c("NN2", "OK2", "UK2")){
		metodo = sub("2", "", metodo)
		result = switch(metodo,
			NN={result=MDI_nearestneighbour2(observaciones,
				ubicaciones)},
			OK={result=MDI_kriging2(observaciones, ubicaciones,
				metodo)},
			UK={result=MDI_kriging2(observaciones, ubicaciones,
				metodo)})}
	else {
		result = switch(metodo,
			NN={result=MDI_nearestneighbour(observaciones,
				ubicaciones)},
			OK={result=MDI_kriging(observaciones, ubicaciones,
				metodo)},
			UK={result=MDI_kriging(observaciones, ubicaciones,
				metodo)})}
	# Terminar y devolver resultados:
	return(result)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_nearestneighbour %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_nearestneighbour = function(observaciones, ubicaciones){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp", lib.loc=libpath)
		require("FNN", lib.loc=libpath)
	})
	# Leer nombres de variables:
	predictores = c("Latitud", "Longitud", "Altitud", "distCosta")
	variables = names(observaciones)[!names(observaciones)
		%in% c("Id", "Codigo", predictores)]
	# Calcular puntos cercanos:
	NN = setNames(data.frame(ubicaciones$Id,
		observaciones$Codigo[FNN::knnx.index(data=observaciones@coords,
		query=ubicaciones@coords, k=1)]), c("Id", "Codigo"))
	# Asignar valores a ubicaciones:
	estimaciones = sp::merge(ubicaciones, NN, by="Id", all.x=TRUE,
		all.y=FALSE, sort=FALSE)
	estimaciones = sp::merge(estimaciones, observaciones[,
		c("Codigo", variables)], by="Codigo", sort=FALSE)
	# Terminar y devolver resultados:
	return(estimaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_nearestneighbour2 %
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MDI_nearestneighbour2 = function(observaciones, ubicaciones){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp", lib.loc=libpath)
		require("FNN", lib.loc=libpath)
	})
	# Leer nombres de variables:
	predictores = c("Latitud", "Longitud", "Altitud", "distCosta")
	variables = names(observaciones)[!names(observaciones)
		%in% c("Id", "Codigo", predictores)]
	# Calcular puntos cercanos:
	NN2 = setNames(data.frame(ubicaciones$Id,
		observaciones$Codigo[FNN::knnx.index(data=observaciones@coords,
		query=ubicaciones@coords, k=2)[,2]]), c("Id", "Codigo"))
	# Asignar valores a ubicaciones:
	estimaciones = sp::merge(ubicaciones, NN2, by="Id", all.x=TRUE,
		all.y=FALSE, sort=FALSE)
	estimaciones = sp::merge(estimaciones,observaciones[,
		c("Codigo", variables)], by="Codigo", sort=FALSE)
	# Terminar y devolver resultados:
	return(estimaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_kriging %
#%%%%%%%%%%%%%%%%%%%%%%%
MDI_kriging = function(observaciones, ubicaciones, metodo="UK"){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp", lib.loc=libpath)
		require("gstat", lib.loc=libpath)
		require("automap", lib.loc=libpath)
	})
	# Variables, predictores y modelos:
	predictores = c("Latitud", "Longitud", "Altitud", "distCosta")
	predic = switch(metodo, OK=1, UK=predictores)
	modelos = vgm()$short[!vgm()$short %in% c("Pow")]
	variables = names(observaciones)[!names(observaciones)
		%in% c("Id", "Codigo", predictores)]
	# Bucle de variables:
	estimaciones = setNames(cbind(data.frame(ubicaciones$Id),
		do.call(cbind, lapply(variables, function(var){tryCatch({
			# Comprobación de campo constante:
			minval = min(observaciones@data[,var])
			maxval = max(observaciones@data[,var])
			if (minval==maxval){return(rep(minval,
				times=nrow(ubicaciones)))}
			# Filtrado de NAs:
			datos = observaciones[!is.na(observaciones@data[,var]),
				c(predictores,var)]
			# Ajuste de variograma:
			Kriformula = as.formula(paste0(var,"~",
				paste(predic, collapse="+")))
			variograma = automap::autofitVariogram(Kriformula, datos,
				model=modelos, debug.level=0)
			# Predicción en ubicaciones:
			estimaciones = gstat::krige(formula=Kriformula,
				locations=datos, newdata=ubicaciones,
				model=variograma$var_model, nmax=Inf, 
				debug.level=0)$var1.pred
			# Corrección de signo inconsistente:
			if (all(observaciones@data[,var] <= 0)){
				estimaciones[estimaciones>0] = 0}
			if (all(observaciones@data[,var] >= 0)){
				estimaciones[estimaciones<0] = 0}
			# Devolver resultados:
			return(estimaciones)},
			error=function(e){return(rep(NA,
				times=nrow(ubicaciones)))})
	}))), c("Id", variables))
	# Asignar valores a ubicaciones:
	estimaciones = sp::merge(ubicaciones, estimaciones, by="Id",
		all.x=TRUE, all.y=FALSE, sort=FALSE)
	# Terminar y devolver resultados:
	return(estimaciones)
}
#%%%%%%%%%%%%%%%%%%%%%%%%
#% Función MDI_kriging2 %
#%%%%%%%%%%%%%%%%%%%%%%%%
MDI_kriging2 = function(observaciones, ubicaciones, metodo="UK"){
	# Cargar scripts externos:
	source("")
	# Cargar variables globales:
	libpath = MDIXX_GlobalData$Directorios$Ruta_Lib
	# Cargar paquetes externos:
	suppressPackageStartupMessages({
		require("sp", lib.loc=libpath)
		require("FNN", lib.loc=libpath)
		require("gstat", lib.loc=libpath)
		require("automap", lib.loc=libpath)
	})
	# Variables, predictores y modelos:
	predictores = c("Latitud", "Longitud", "Altitud", "distCosta")
	predic = switch(metodo, OK=1, UK=predictores)
	modelos = vgm()$short[!vgm()$short %in% c("Pow")]
	variables = names(observaciones)[!names(observaciones)
		%in% c("Id", "Codigo", predictores)]
	# Calcular puntos cercanos:
	NN = setNames(data.frame(ubicaciones$Id,
		observaciones$Codigo[FNN::knnx.index(data=observaciones@coords,
		query=ubicaciones@coords,k=1)]), c("Id", "Codigo"))
	# Bucle de variables:
	estimaciones = setNames(cbind(data.frame(ubicaciones$Id),
		do.call(cbind, lapply(variables,function(var){
		# Bucle de ubicaciones:
		do.call(c, lapply(NN$Id,function(id){
			ubi = ubicaciones[ubicaciones$Id==id,]
			omi = NN[NN$Id==id,"Codigo"]
			obs = observaciones[!observaciones$Codigo==omi,]
			estimaciones = tryCatch({
				# Comprobación de campo constante:
				minval = min(obs@data[,var])
				maxval = max(obs@data[,var])
				if (minval==maxval){return(rep(minval,
					times=nrow(ubi)))}
				# Filtrado de NAs:
				datos = obs[!is.na(obs@data[,var]),
					c(predictores, var)]
				# Ajuste de variograma:
				Kriformula = as.formula(paste0(var, "~",
					paste(predic, collapse="+")))
				variograma = automap::autofitVariogram(Kriformula,
					datos, model=modelos, debug.level=0)
				# Predicción en ubicaciones:
				estimaciones = gstat::krige(formula=Kriformula,
					locations=datos, newdata=ubi,
					model=variograma$var_model, nmax=Inf,
					debug.level=0)$var1.pred
				# Corrección de signo inconsistente:
				if (all(observaciones@data[,var] <= 0)){
					estimaciones[estimaciones>0] = 0}
				if (all(observaciones@data[,var] >= 0)){
					estimaciones[estimaciones<0] = 0}
				# Devolver resultados:
				return(estimaciones)},
				error=function(e){print(e); return(rep(NA,
					times=nrow(ubi)))})
			# Devolver resultados:
			return(estimaciones)}))
	}))), c("Id", variables))
	# Asignar valores a ubicaciones:
	estimaciones = sp::merge(ubicaciones,estimaciones, by="Id",
		all.x=TRUE, all.y=FALSE, sort=FALSE)
	# Terminar y devolver resultados:
	return(estimaciones)
}