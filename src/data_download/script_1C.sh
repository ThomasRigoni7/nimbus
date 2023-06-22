# data.json contains username, password (maybe), and date range
# geojson file contains a rough region of interest (using too many points breaks the API)
# If data.json is not stored in a private place, the password field can be left empty and be provided using the command line
python sentinel_download_1C.py --data data.json --geojson switzerland_blob.geojson
