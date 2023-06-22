# data.json contains username, password (maybe), and date range
# geojson file contains a rough region of interest (using too many points breaks the API)
# If data.json is not stored in a private place, the password field can be left empty and be provided using the command line


for ATTEMPT in {1..48}
do
    echo "[Iteration $ATTEMPT]"
    python sentinel_download_1C.py --outdir ./raw_data_32TNS_1C/ --data ~/data_32TNS.json --geojson ~/area_32TNS.geojson
    echo "Sleeping for 30 minutes..."
    sleep 30m
done
