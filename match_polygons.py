import fiona
from shapely.geometry import Polygon, shape, mapping
import rtree
import csv

# Open each layer
poly_layer_dian = fiona.open('Dataset_ktimatologio_Herakleion_new_pst/dian.shp')
poly_layer_pst = fiona.open('Dataset_ktimatologio_Herakleion_new_pst/pst.shp')

#print(poly_layer_pst)
# Convert to lists of shapely geometries
dian_polygons = [shape(dian['geometry']) for dian in poly_layer_dian]
pst_polygons = [(poly['id'], shape(poly['geometry'])) for poly in poly_layer_pst]

dian_polygons_ids = [dian['id'] for dian in poly_layer_dian]

#print(dian_polygons)
#print(pst_polygons)

# Create spatial index
spatial_index = rtree.index.Index()
for idx, poly_tuple in enumerate(pst_polygons):
#	print(poly_tuple)
	_, poly = poly_tuple
	spatial_index.insert(idx, poly.bounds)

schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int', 'original_i': 'int', 'descr': 'str'},
}

coordinates_list = []

# Find intersections
id = 0
count = 0
opened_file = 0
covered_polygons = {}
#geometry_list = []

#print(dian_polygons[:20])

f = open("polygon_data.csv", "w")
writer = csv.writer(f)

for polygon, polygon_id in zip(dian_polygons, dian_polygons_ids):
	
	if polygon.geom_type == 'Polygon':
		if polygon.exterior.coords.xy not in coordinates_list:
			coordinates_list.append(polygon.exterior.coords.xy)

			# Write a new Shapefile
			#if id > 0 and id % 20 == 0:
			#	print(id)
			#	count += 1
			#	opened_file = 0
			
			if not opened_file:
				"""
				with fiona.open('my_shp.shp', 'w', 'ESRI Shapefile', schema) as c:
					## If there are multiple geometries, put the "for" loop here
					
					c.write({
						'geometry': mapping(polygon),
						'properties': {'id': id, 'original_i': polygon_id, 'descr': 'dian'},
					})
				"""
				opened_file = 1
			else:
				"""
				with fiona.open('my_shp.shp', 'a', 'ESRI Shapefile', schema) as c:
					## If there are multiple geometries, put the "for" loop here
					
					c.write({
						'geometry': mapping(polygon),
						'properties': {'id': id, 'original_i': polygon_id, 'descr': 'dian'},
					})
				"""
			
			for idx in list(spatial_index.intersection(polygon.bounds)):
				if pst_polygons[idx][1].intersects(polygon):
					if pst_polygons[idx][1].area > polygon.area:
						if (pst_polygons[idx][1].intersection(polygon).area/pst_polygons[idx][1].area) >= 0.9:
							
							writer.writerows([[polygon, pst_polygons[idx][1]]])
							#print(pst_polygons[idx][1])
							"""
							with fiona.open('my_shp.shp', 'a', 'ESRI Shapefile', schema) as c:
								c.write({
									'geometry': mapping(pst_polygons[idx][1]),
									'properties': {'id': id, 'original_i': pst_polygons[idx][0], 'descr': 'pst'},
								})
							"""
							if idx not in covered_polygons.keys():
								covered_polygons[idx] = [polygon]
							else:
								covered_polygons[idx].append(polygon)
							
					else:
						if (pst_polygons[idx][1].intersection(polygon).area/polygon.area) >= 0.9:
							row = {}
							#row['dian_poly'] = polygon
							#row['pst_poly'] = pst_polygons[idx][1]
							#writer.writerows(row)
							writer.writerows([[polygon, pst_polygons[idx][1]]])
							#print(pst_polygons[idx][1])
							"""
							with fiona.open('my_shp.shp', 'a', 'ESRI Shapefile', schema) as c:
								c.write({
									'geometry': mapping(pst_polygons[idx][1]),
									'properties': {'id': id, 'original_i': pst_polygons[idx][0], 'descr': 'pst'},
								})
							"""
							if idx not in covered_polygons.keys():
								covered_polygons[idx] = [polygon]
							else:
								covered_polygons[idx].append(polygon)
			id += 1

f.close()
