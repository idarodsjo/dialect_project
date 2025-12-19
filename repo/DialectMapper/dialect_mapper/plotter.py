import json
import matplotlib as mpl
import matplotlib.colors as mpl_colors
import os
import re
from shapely import Polygon, MultiPolygon, affinity

import math
from numpy import log as ln

import cairosvg

import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import mapping_data

class ColorMap():
    '''
    This class is copied/borrowed from the geoplotlib package
    '''
    def __init__(self, cmap_name, levels=10):
        """
        Converts continuous values into colors using matplotlib colorscales
        :param cmap_name: colormap name
        :param levels: discretize the colorscale into levels
        """
        self.cmap = mpl.colormaps[cmap_name]
        self.levels = levels

    def to_color_linear_scale(self, value, maxvalue, minvalue=0.0):
        """
        convert continuous values into colors using matplotlib colorscales
        :param value: value to be converted
        :param maxvalue: max value in the colorscale
        :param minvalue: minimum of the input values in linear scale (default is 0)
        :return: the color corresponding to the value
        """
        if minvalue >= maxvalue:
            raise Exception('minvalue must be less than maxvalue')
        else:
            value = 1.*(value-minvalue) / (maxvalue-minvalue)

        if value < 0:
            value = 0
        elif value > 1:
            value = 1

        value = int(1.*self.levels*value)*1./(self.levels-1)
        value = self.cmap(value)
        return mpl_colors.to_hex(value)



class plotter_methods:
    '''
    A class to make creating maps/plots of Norway easy. Of especial use is shading different regions
    '''

    def _convert_latlon_to_xy(self, latitude, longitude, mapWidth=200, mapHeight=100, move_south=False):
        """
        The geoJSON format keeps points in latitude and longitude format which is fine
        But if we try to plot the raw values it makes Norway look very stretched east-west
        Therefore we want a way of converting them into a more standard projection
        We're uusing the Mercator projection
        Implementation copied from https://stackoverflow.com/questions/14329691/convert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection
        """
        if move_south:
            latitude = latitude - self.latitude_southern_adjustment
            longitude = longitude - self.longitude_southern_adjustment

        x = (longitude + 180) * (mapWidth / 360)

        # convert from degrees to radians
        latRad = (latitude * math.pi) / 180

        # get y value
        mercN = ln(math.tan((math.pi / 4) + (latRad / 2)))
        y     = (mapHeight / 2) - (mapWidth * mercN / (2 * math.pi))
        
        return x, y

    def _create_poly(self, obj, final_width, final_height, move_south=False):
        # convert all lat/lon pairs into x and y 
        obj = [[self._convert_latlon_to_xy(pair[1], pair[0], mapWidth=final_width, mapHeight=final_height, move_south=move_south) for pair in shape] for shape in obj]
        if len(obj) > 1:
            return Polygon(obj[0], holes=obj[1:])
        else:
            return Polygon(obj[0])

    def _process_features(self, features, get_color, final_width, final_height, split_norway=False, rotate_norway=False, stroke_width=0.025):
        svg_list = []
        min_x = None
        min_y = None
        max_x = None
        max_y = None
        for region_features in features:
            if split_norway and region_features['properties']['navn'] in self.northern_regions:
                region_multiPolygon = MultiPolygon(
                    [self._create_poly(obj, final_width, final_height, move_south=True) for obj in region_features['geometry']['coordinates']]
                )
            else:    
                region_multiPolygon = MultiPolygon(
                    [self._create_poly(obj, final_width, final_height) for obj in region_features['geometry']['coordinates']]
                )
                if rotate_norway:
                    region_multiPolygon = affinity.rotate(region_multiPolygon, -30, origin=(0, 9))
            region_name = region_features['properties']['navn']
            # ugly way of dealing with our 1 problematic kommune
            if 'Herøy' in region_name:
                if region_features['geometry']['coordinates'][0][0][0][0] == 12.277040240769484:
                    region_name += '_Helgelandsk'
                else:
                    region_name += '_Nordvestlandsk'
            if get_color(region_name):
                svg_list.append(
                    self.stroke_width_pat.sub(
                        'stroke-width="{}"'.format(str(stroke_width)),
                        region_multiPolygon.svg(fill_color=get_color(region_name), opacity=1)
                    )   
                )
            else:
                svg_list.append(
                    self.stroke_width_pat.sub(
                        'stroke-width="{}"'.format(str(stroke_width)),
                        region_multiPolygon.svg(fill_color='#ffffff', opacity=1)
                    )   
                )
            mp_bounds = region_multiPolygon.bounds
            if min_x == None:
                min_x = mp_bounds[0]
            if min_y == None:
                min_y = mp_bounds[1]
            if max_x == None:
                max_x = mp_bounds[2]
            if max_y == None:
                max_y = mp_bounds[3]
            
            if mp_bounds[0] < min_x:
                min_x = mp_bounds[0]
            if mp_bounds[1] < min_y:
                min_y = mp_bounds[1]
            if mp_bounds[2] > max_x:
                max_x = mp_bounds[2]
            if mp_bounds[3] > max_y:
                max_y = mp_bounds[3]

        width = max_x - min_x
        height = max_y - min_y
        return svg_list, min_x, min_y, width, height

    def _save_output(
        self,
        output_path: str,
        final_width: int,
        final_height: int,
        min_x: float,
        min_y: float,
        width: float,
        height: float,
        svg_list: list):
        
        output_png = False
        output_pdf = False
        if output_path[-4:] == ".png":
            output_svg_filepath = output_path[:-4] + ".svg"
            output_png = True
        elif output_path[-4:] == ".pdf":
            output_svg_filepath = output_path[:-4] + ".svg"
            output_pdf = True
        else:
            output_svg_filepath = output_path
        
        with open(output_svg_filepath, 'w') as open_f:
            open_f.write(
                self.head_bit.format(
                    str(final_width),
                    str(final_height),
                    min_x, 
                    min_y, 
                    width, 
                    height ) + 
                ''.join(svg_list) + 
                self.end_bit
            )
        
        if output_png:
            cairosvg.svg2png(url=output_svg_filepath, write_to=output_path)
            os.remove(output_svg_filepath)
        if output_pdf:
            cairosvg.svg2pdf(url=output_svg_filepath, write_to=output_path)
            os.remove(output_svg_filepath)
    
    def plot_kommune_regions(
        self, 
        output_svg_filepath, 
        kommune_region_to_value={}, 
        color_map_name='Blues', 
        color_map_levels=50, 
        max_region_value=30, 
        default_color='#66cc99', 
        final_width='500', 
        final_height='500'):

        final_width = float(final_width)
        final_height = float(final_height)
        cmap = ColorMap(color_map_name, levels=color_map_levels)
        def get_color(dialect_name):
            if dialect_name in kommune_region_to_value:
                return cmap.to_color_linear_scale(kommune_region_to_value.get(dialect_name), max_region_value)
            else:
                return default_color
        svg_list, min_x, min_y, width, height = self._process_features(
            self.kommuner_json['features'], 
            get_color,
            final_height,
            final_width
            )
        self._save_output(
            output_svg_filepath,
            final_width, 
            final_height,
            min_x, 
            min_y, 
            width, 
            height,
            svg_list
        )

    def plot_card4_dialect_regions(
        self, 
        output_svg_filepath, 
        dia_region_to_value={}, 
        color_map_name='Blues', 
        color_map_levels=50, 
        max_region_value=30, 
        default_color='#66cc99', 
        final_width='500', 
        final_height='500'):

        final_width = float(final_width)
        final_height = float(final_height)
        cmap = ColorMap(color_map_name, levels=color_map_levels)
        def get_color(dialect_name):
            if dialect_name in dia_region_to_value:
                if dia_region_to_value.get(dialect_name) == None:
                    return None
                else:
                    return cmap.to_color_linear_scale(dia_region_to_value.get(dialect_name), max_region_value)
            else:
                return default_color
        svg_list, min_x, min_y, width, height = self._process_features(
            self.card4_dialekter_json['features'], 
            get_color,
            final_height,
            final_width
            )
        self._save_output(
            output_svg_filepath,
            final_width, 
            final_height,
            min_x, 
            min_y, 
            width, 
            height,
            svg_list
        )

    def plot_card5_dialect_regions(
        self, 
        output_svg_filepath, 
        dia_region_to_value={}, 
        color_map_name='Blues', 
        color_map_levels=50, 
        max_region_value=30, 
        default_color='#66cc99', 
        final_width='500', 
        final_height='500'):

        final_width = float(final_width)
        final_height = float(final_height)
        cmap = ColorMap(color_map_name, levels=color_map_levels)
        def get_color(dialect_name):
            if dialect_name in dia_region_to_value:
                if dia_region_to_value.get(dialect_name) == None:
                    return None
                else:
                    return cmap.to_color_linear_scale(dia_region_to_value.get(dialect_name), max_region_value)
            else:
                return default_color
        svg_list, min_x, min_y, width, height = self._process_features(
            self.card5_dialekter_json['features'], 
            get_color,
            final_height,
            final_width
            )
        self._save_output(
            output_svg_filepath,
            final_width, 
            final_height,
            min_x, 
            min_y, 
            width, 
            height,
            svg_list
        )

    def plot_dialect_regions(
        self, 
        output_svg_filepath, 
        dialect_region_to_value={}, 
        color_map_name='Blues', 
        color_map_levels=50, 
        max_region_value=30, 
        default_color='#66cc99', 
        final_width='500', 
        final_height='500'):

        cmap = ColorMap(color_map_name, levels=color_map_levels)
        def get_color(dialect_name):
            if dialect_name in dialect_region_to_value:
                return cmap.to_color_linear_scale(dialect_region_to_value.get(dialect_name), max_region_value)
            else:
                return default_color

        final_width = float(final_width)
        final_height = float(final_height)
        svg_list, min_x, min_y, width, height = self._process_features(
            self.dialekter_json['features'], 
            get_color,
            final_width,
            final_height
        )
        self._save_output(
            output_svg_filepath,
            final_width, 
            final_height,
            min_x,
            min_y, 
            width, 
            height,
            svg_list
        )

    def plot_rundkast_regions(
        self, 
        output_svg_filepath, 
        rundkast_region_to_value={}, 
        color_map_name='Blues', 
        color_map_levels=50, 
        max_region_value=30, 
        default_color='#66cc99', 
        final_width='500', 
        final_height='500',
        split_norway=False,
        rotate_norway=False,
        stroke_width=0.025
    ):

        cmap = ColorMap(color_map_name, levels=color_map_levels)
        def get_color(region_name):
            if region_name in rundkast_region_to_value:
                return cmap.to_color_linear_scale(rundkast_region_to_value.get(region_name), max_region_value)
            else:
                return default_color

        final_width = float(final_width)
        final_height = float(final_height)
        svg_list, min_x, min_y, width, height = self._process_features(
            self.region_json['features'], 
            get_color,
            final_width,
            final_height,
            split_norway=split_norway,
            rotate_norway=rotate_norway,
            stroke_width=stroke_width
        )
        self._save_output(
            output_svg_filepath,
            final_width, 
            final_height,
            min_x, 
            min_y, 
            width, 
            height,
            svg_list
        )
        
    def __init__(self) -> None:
        ### Original geoJSON data from https://github.com/robhop/fylker-og-kommuner-2020
        self.dialekter_json = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'dialekter_geojson.json'
                )
            )
        )
        self.card4_dialekter_json = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'card4_region_geojson.json'
                )
            )
        )
        self.card5_dialekter_json = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'card5_region_geojson.json'
                )
            )
        )
        self.kommuner_json = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'kommuner_komprimert.json'
                )
            )
        )
        self.region_json = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'rundkast_regions_geojson.json'
                )
            )
        )
        self.northern_regions = json.load(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'northern_region.json'
                )
            )
        )
        self.northern_regions = self.northern_regions['northern_regions']
        self.latitude_southern_adjustment = 2.75 # 3.2 these also work but top have is lower and further west
        self.longitude_southern_adjustment = 10 # 12
        self.stroke_width_pat = re.compile('stroke-width=".*?"')
        self.head_bit = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}" height="{}" viewBox="{} {} {} {}" preserveAspectRatio="xMinYMin meet">'''
        self.end_bit = '''</svg>'''
