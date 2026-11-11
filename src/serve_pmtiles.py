# ************************************************************
# LISFLOOD-FP to FIM
# Utility - serve_pmtiles
#
# Serves the PMTiles from build_pmtiles_06 as XYZ tiles on localhost so
# they can be viewed in QGIS (or any XYZ client).  GDAL / QGIS can not open
# raster PMTiles directly (the GDAL PMTiles driver reads vector tiles only).
#
# Every *.pmtiles in the folder is served at
#     http://localhost:<port>/<file name without .pmtiles>/{z}/{x}/{y}.png
# and http://localhost:<port>/ lists the files and their XYZ URLs.
#
# In QGIS: Browser > XYZ Tiles > New Connection, paste the URL, set the zoom
# range (3 to 15) and set Interpretation = "Terrarium Terrain RGB" so the
# layer holds depth in meters (dry = 0) instead of RGB colors.
#
# Requires: pmtiles
#
# Created by: Andy Carter, PE
# Created - 2026.11.10
# ************************************************************

# ************************************************************
import os
import glob
import re

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pmtiles.reader import Reader, MmapSource

import argparse
# ************************************************************


# --------------------------------------------------
def fn_open_readers(str_folder):
    # {name without .pmtiles: (Reader, open file)} for every *.pmtiles
    dict_readers = {}
    for str_path in sorted(glob.glob(os.path.join(str_folder, '*.pmtiles'))):
        str_name = os.path.splitext(os.path.basename(str_path))[0]
        file_in = open(str_path, 'rb')
        dict_readers[str_name] = (Reader(MmapSource(file_in)), file_in)
    return dict_readers
# --------------------------------------------------


# --------------------------------------------------
def fn_make_handler(dict_readers, int_port):

    regex_tile = re.compile(r'^/([^/]+)/(\d+)/(\d+)/(\d+)\.png$')

    class TileHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            match = regex_tile.match(self.path)
            if match and match.group(1) in dict_readers:
                reader = dict_readers[match.group(1)][0]
                int_z, int_x, int_y = (int(match.group(i)) for i in (2, 3, 4))
                bytes_tile = reader.get(int_z, int_x, int_y)
                if bytes_tile is None:
                    # dry tiles are not stored
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(bytes_tile)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(bytes_tile)
                return

            if self.path in ('/', '/index.html'):
                list_lines = ['<html><body><h3>PMTiles (Terrarium depth)</h3><ul>']
                for str_name in dict_readers:
                    str_url = f'http://localhost:{int_port}/{str_name}/{{z}}/{{x}}/{{y}}.png'
                    list_lines.append(f'<li>{str_name}<br><code>{str_url}</code></li>')
                list_lines.append('</ul></body></html>')
                bytes_body = '\n'.join(list_lines).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(bytes_body)))
                self.end_headers()
                self.wfile.write(bytes_body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass    # quiet; one line per tile is too much

    return TileHandler
# --------------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='========= SERVE PMTILES AS XYZ TILES =========')

    parser.add_argument('-i',
                        dest = "str_folder",
                        help=r'REQUIRED: folder of .pmtiles Example: output/121002030402/06_pmtiles',
                        required=True,
                        metavar='DIR')

    parser.add_argument('-p',
                        dest = "int_port",
                        help=r'OPTIONAL: port Default: 8000',
                        required=False,
                        default=8000,
                        type=int)

    args = vars(parser.parse_args())

    dict_readers = fn_open_readers(args['str_folder'])
    if not dict_readers:
        raise FileNotFoundError(f"No .pmtiles found in {args['str_folder']}")

    print(f" -- Serving {len(dict_readers)} PMTiles from {os.path.abspath(args['str_folder'])}")
    for str_name in dict_readers:
        print(f"    http://localhost:{args['int_port']}/{str_name}/{{z}}/{{x}}/{{y}}.png")
    print(" -- Ctrl+C to stop")

    server = ThreadingHTTPServer(('127.0.0.1', args['int_port']),
                                 fn_make_handler(dict_readers, args['int_port']))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for reader, file_in in dict_readers.values():
            file_in.close()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
