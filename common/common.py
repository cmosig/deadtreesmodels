import utm
import numpy as np
import json
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform
import shapely
import geopandas as gpd
import rasterio
import rasterio.warp
import rasterio.enums
from rasterio.vrt import WarpedVRT

def get_utm_string_from_latlon(lat, lon):
    zone = utm.from_latlon(lat, lon)
    utm_code = 32600 + zone[2]
    if lat < 0:
        utm_code -= 100

    return f"EPSG:{utm_code}"


def merge_polygons(contours, hierarchy) -> MultiPolygon:
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # hierarchy structure: [next, prev, first_child, parent]

    def make_valid(polygon):
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon

    polygons = []

    idx = 0
    while idx != -1:
        # Get contour from global list of contours
        contour = np.squeeze(contours[idx])

        # cv2.findContours() sometimes returns a single point -> skip this case
        if len(contour) > 2:
            # Convert contour to shapely polygon
            holes = []

            # check if there is a child
            child_idx = hierarchy[idx][2]
            if child_idx != -1:
                # iterate over all children and add them as holes
                while child_idx != -1:
                    child = np.squeeze(contours[child_idx])
                    if len(child) > 2:
                        holes.append(child)
                    child_idx = hierarchy[child_idx][0]

            new_poly = Polygon(shell=contour, holes=holes)

            # save poly
            polygons.append(new_poly)

        # Check if there is some next polygon at the same hierarchy level
        idx = hierarchy[idx][0]

    return polygons


def mask_to_polygons(mask, dataset_reader):
    """
    this function takes a numpy mask as input and returns a list of polygons
    that are in the crs of the passed dataset reader
    """

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8).copy(),
                                           mode=cv2.RETR_CCOMP,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    # if no contours are found, return empty list
    if hierarchy is None or len(hierarchy) == 0:
        return []

    hierarchy = hierarchy[0]

    poly = merge_polygons(contours, hierarchy)

    # affine transform from pixel to world coordinates
    transform = dataset_reader.transform
    transform_matrix = (transform.a, transform.b, transform.d, transform.e,
                        transform.c, transform.f)
    poly = [affine_transform(p, transform_matrix) for p in poly]

    return poly


def save_poly(filename, poly, crs):
    """saves the polygons to a file in the given crs"""
    gpd.GeoDataFrame(geometry=poly, crs=crs).to_file(filename)


def image_reprojector(input_tif, min_res=0, max_res=1e9):
    dataset = rasterio.open(input_tif)
    centroid = dataset.lnglat()
    utm_crs = get_utm_string_from_latlon(centroid[1], centroid[0])

    default_transform, width, height = rasterio.warp.calculate_default_transform(
        dataset.crs, utm_crs, dataset.width, dataset.height, *dataset.bounds)

    # get original resolution
    orig_res = default_transform.a

    target_res = None
    if orig_res < min_res:
        target_res = min_res
        print(
            f"Original resolution ({orig_res}) is smaller than minimum resolution ({min_res}). Reprojecting to minimum resolution."
        )
    if orig_res > max_res:
        target_res = max_res
        print(
            f"Original resolution ({orig_res}) is larger than maximum resolution ({max_res}). Reprojecting to maximum resolution."
        )

    if target_res is not None:
        default_transform, width, height = rasterio.warp.calculate_default_transform(
            dataset.crs,
            utm_crs,
            dataset.width,
            dataset.height,
            *dataset.bounds,
            resolution=target_res)
        
    vrt = WarpedVRT(dataset, crs=utm_crs, transform=default_transform, width=width, height=height, dtype="uint8", nodata=0)

    return vrt


def reproject_polygons(polygons, src_crs, dst_src):
    """
    reprojects the polygons from the src_crs to the dst_crs
    """
    rep = rasterio.warp.transform_geom(src_crs, dst_src, polygons)
    if isinstance(rep, list):
        return shapely.from_geojson([json.dumps(x) for x in rep])
    else:
        return shapely.from_geojson(json.dumps(rep))


def filter_polygons_by_area(polygons, min_area):
    """
    filters the polygons by the minimum area
    """
    filtered = []
    for p in polygons:

        exterior = p.exterior

        # Filter holes (interior rings) by area
        filtered_holes = [hole for hole in p.interiors
                         if Polygon(hole).area >= min_area]

        # Create new polygon with filtered holes
        filtered_p = Polygon(exterior, filtered_holes)

        if filtered_p.area >= min_area:
            filtered.append(p)

    print(f"Filtered {len(polygons) - len(filtered)} polygons by minimum area of {min_area}m2.")
    return filtered