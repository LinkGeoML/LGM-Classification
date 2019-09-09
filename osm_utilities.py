import pandas as pd
import re
import json
import requests
from shapely.geometry import LineString, Polygon


def query_osm_data(query, fpath):
    overpass_url = 'http://overpass-api.de/api/interpreter'
    response = requests.get(overpass_url, params={'data':query}).json()
    with open(fpath, 'w') as f:
        json.dump(response, f)
    return


def parse_osm_streets(fpath):
    # Helper function
    def convert_to_wkt_geometry(row):
        lons = [p['lon'] for p in row['geometry']]
        lats = [p['lat'] for p in row['geometry']]
        return LineString(list(zip(lons, lats)))

    with open(fpath, encoding='utf-8') as f:
        streets = json.load(f)['elements']

    data = [(street['id'], street['geometry']) for street in streets]
    cols = ['id', 'geometry']
    street_df = pd.DataFrame(data=data, columns=cols)
    street_df['geometry'] = street_df.apply(convert_to_wkt_geometry, axis=1)
    return street_df


def download_osm_streets(bbox_coords, exp_path):
    fpath = exp_path + '/osm_streets.json'
    query = (
        '[out:json]'
        f'[bbox:{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}];'
        'way["highway"];'
        'out geom;')
    query_osm_data(query, fpath)
    street_df = parse_osm_streets(fpath)
    fpath = exp_path + '/osm_streets.csv'
    street_df.to_csv(f'{fpath}', columns=['id', 'geometry'], index=False)
    return


def parse_osm_polys(fpath):
    # Helper function
    def extract_name_tags(row):
        names = [tag[1] for tag in row['tags'] if re.search('name', tag[0])]
        return names

    # Helper function
    def convert_to_wkt_geometry(row):
        lons = [p['lon'] for p in row['geometry']]
        lats = [p['lat'] for p in row['geometry']]
        return Polygon(list(zip(lons, lats)))

    with open(fpath, encoding='utf-8') as f:
        polys = json.load(f)['elements']

    data = []
    for poly in polys:
        if 'tags' in poly:
            poly_tags = [(k, v) for k, v in poly['tags'].items()]
            data.append((poly['id'], poly_tags, poly['geometry']))

    cols = ['id', 'tags', 'geometry']
    poly_df = pd.DataFrame(data=data, columns=cols)
    poly_df['name'] = poly_df.apply(extract_name_tags, axis=1)
    poly_df['geometry'] = poly_df.apply(convert_to_wkt_geometry, axis=1)
    return poly_df


def download_osm_polygons(bbox_coords, exp_path):
    fpath = exp_path + '/osm_polys.json'
    query = (
        '[out:json]'
        f'[bbox:{bbox_coords[0]},{bbox_coords[1]},{bbox_coords[2]},{bbox_coords[3]}];'
        'way(if:is_closed());'
        'out geom;')
    query_osm_data(query, fpath)
    poly_df = parse_osm_polys(fpath)
    fpath = exp_path + '/osm_polys.csv'
    cols = ['id', 'name', 'tags', 'geometry']
    poly_df.to_csv(f'{fpath}', columns=cols, index=False)
    return
