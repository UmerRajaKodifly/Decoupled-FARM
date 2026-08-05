# Data

`construction_site_object_vocabulary.json` is a snapshot of the Spatial GPT
construction-site object list, vendored here so Decoupled FARM does not read
another checkout. The adapter in `src/farm_object_map/vocab.py` turns it into
FARM’s one-name-per-line vocab text at runtime.
