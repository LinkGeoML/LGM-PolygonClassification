import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry import shape
import shapely
import shapely.wkt
import fiona
import numpy as np

def get_X_Y_data(filename):
	df = pd.read_csv(filename)

	features_dian = []
	features_pst = []
	#labels = []
	for index, row in df.iterrows():
		feature_list_dian = []
		feature_list_pst = []
		
		dian_shape = shapely.wkt.loads(row[0])
		pst_shape = shapely.wkt.loads(row[1])
		
		# get area features
		area_dian = dian_shape.area
		area_pst = pst_shape.area
		
		# get intersection percentage features
		intersection_area_dian = dian_shape.intersection(pst_shape).area/dian_shape.area
		intersection_area_pst = pst_shape.intersection(dian_shape).area/pst_shape.area
		
		# get perimeter features
		perimeter_dian = dian_shape.length
		perimeter_pst = pst_shape.length
		
		# get vertices features
		dian_vertex_count = len(list(dian_shape.exterior.coords))
		pst_vertex_count = len(list(pst_shape.exterior.coords))
		
		dian_coords = list(dian_shape.exterior.coords)
		len_list = []
		for i in range(0, len(dian_coords)-1):
			point1 = Point(dian_coords[i])
			point2 = Point(dian_coords[i+1])
			
			length = point1.distance(point2)
			len_list.append(length)
		
		len_array_dian = np.asarray(len_list)
		mean_dian = np.mean(len_array_dian)
		var_dian = np.var(len_array_dian)
			
		pst_coords = list(pst_shape.exterior.coords)
		len_list = []
		for i in range(0, len(dian_coords)-1):
			point1 = Point(dian_coords[i])
			point2 = Point(dian_coords[i+1])
			
			length = point1.distance(point2)
			len_list.append(length)
		
		len_array_pst = np.asarray(len_list)
		mean_pst = np.mean(len_array_pst)
		var_pst = np.var(len_array_pst)
		
		feature_list_dian = [area_dian, intersection_area_dian, perimeter_dian, dian_vertex_count, mean_dian, var_dian]
		feature_list_pst = [area_pst, intersection_area_pst, perimeter_pst, pst_vertex_count, mean_pst, var_pst]
		
		features_dian.append(feature_list_dian)
		features_pst.append(feature_list_pst)
		
		feature_array_dian = np.asarray(features_dian)
		feature_array_pst = np.asarray(features_pst)
		
		#if var_pst > 9900:
		#	print(row)
	
	"""
	boundaries_mean_dian = np.ones(feature_array_dian.shape[0]) * (np.mean(feature_array_dian[:,3]) - 1)
	boundaries_mean_pst = np.ones(feature_array_pst.shape[0]) * (np.mean(feature_array_dian[:,3]) - 1)

	boundaries_var_dian = np.ones(feature_array_dian.shape[0]) * (np.var(feature_array_dian[:,3]) - 1)
	boundaries_var_pst = np.ones(feature_array_pst.shape[0]) * (np.var(feature_array_dian[:,3]) - 1)

	boundaries_mean_dian = np.reshape(boundaries_mean_dian, (boundaries_mean_dian.shape[0],1))
	boundaries_var_dian = np.reshape(boundaries_var_dian, (boundaries_var_dian.shape[0],1))
	boundaries_mean_pst = np.reshape(boundaries_mean_pst, (boundaries_mean_pst.shape[0],1))
	boundaries_var_pst = np.reshape(boundaries_var_pst, (boundaries_var_pst.shape[0],1))
	

	feature_array_dian = np.concatenate((feature_array_dian, boundaries_mean_dian, boundaries_var_dian), axis = 1)
	feature_array_pst = np.concatenate((feature_array_pst, boundaries_mean_pst, boundaries_var_pst), axis = 1)
	feature_array_dian = np.concatenate((feature_array_dian, boundaries_mean_pst, boundaries_mean_pst), axis = 1)
	feature_array_pst = np.concatenate((feature_array_pst, boundaries_mean_dian, boundaries_mean_dian), axis = 1)
	"""
	feature_array = np.concatenate((feature_array_dian, feature_array_pst), axis = 0)

	dian_labels = np.zeros(feature_array_dian.shape[0])
	pst_labels = np.ones(feature_array_pst.shape[0])

	labels = np.concatenate((dian_labels, pst_labels), axis = 0)
	
	

	print(feature_array.shape)
	print(labels.shape)
	
	return feature_array, labels

	#print(np.mean(feature_array_dian[:,2]))
	#print(np.var(feature_array_dian[:,2]))
	#print(feature_array_dian.shape)
	#print(feature_array_pst.shape)
	
def standardize_data_train(X):
	from sklearn.preprocessing import MinMaxScaler
	
	standard_scaler = MinMaxScaler()
	X = standard_scaler.fit_transform(X)
	
	return X, standard_scaler

def standardize_data_test(X, scaler):
	from sklearn.preprocessing import MinMaxScaler
	
	X = standard_scaler.transform(X)
	
	return X
		
		

