
from zipfile import ZipFile
import json
import pandas as pd

from mem import cache

def json_to_flat_dict(x):
    ''' Convert a json string to a flat python dictionary
    which can be passed into Pandas. '''
    ob = json.loads(x)
    for k, v in ob.items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

@cache
def load_reviews(data_zip_file, training_set_file):
    zf = ZipFile(data_zip_file)
    reviews = zf.open(training_set_file)

    df = pd.DataFrame([json_to_flat_dict(line) for line in reviews])
    return df

