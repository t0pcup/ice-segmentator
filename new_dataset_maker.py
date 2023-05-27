import warnings
from shapely.geometry import Polygon
from shapely.ops import unary_union
import os
import glob
from eodag.api.core import EODataAccessGateway
from eodag import setup_logging
import geopandas as gpd
from tqdm import tqdm
import time
from datetime import date, timedelta
import fiona

# warnings.filterwarnings("ignore")
reg_path = 'C:/files/regions/2020'
workspace = 'C:/files/dag_img_2'

# setup_logging(0)
setup_logging(verbose=3, no_progress_bar=False)
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "###############################"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "###########"

# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__USERNAME"] = ["t0pcup@yandex.ru", "kdmikhaylova_1@edu.hse.ru"][0]
# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__PASSWORD"] = ["###############", "###############"][0]

if not os.path.isdir(workspace):
    os.mkdir(workspace)

yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
    f_yml.write(yaml_content.strip())

dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
product_type = 'S1_SAR_GRD'
dag.set_preferred_provider("peps")


def verify(file: str):
    schema_dataset = fiona.open(file)
    # get water, ice, shelf; ignore land and no data: ['L', 'N']
    # lst, idx = ['W', 'I', 'S'], []
    lst = [10, 12, 13,
           20, 23, 24,
           30, 34, 35,
           40, 45, 46,
           50, 56, 57,
           60, 67, 68,
           70, 78, 79,
           80, 89, 81]
    lst = list(map(str, lst))
    idx = []
    my_ind = -1
    for obj in schema_dataset:
        my_ind += 1
        # if obj['properties']['POLY_TYPE'] in lst:
        if obj['properties']['CT'] in lst:
            idx.append(my_ind)
    return idx


def scroll(pg):
    lst = []
    for el in pg:
        # if {'1SDH', 'EW'} & set(elt.properties["title"].split('_')):
        #     continue
        if {'1SDV', 'IW'} & set(el.properties["title"].split('_')):
            try:
                product_path_ = el.download(extract=False)
                lst.append(product_path_)
            except:
                pass
    return lst


skip = 0
for f in glob.glob(f'{reg_path}/*.shp'):
    cnt, indexes = 0, verify(f)
    if len(indexes) == 0:
        skip += 1
        continue
    dataset = gpd.read_file(f).to_crs('epsg:4326')
    nm = f.split("\\")[-1][4:].split("T")[0][5:7]
    dt = f.split('_')[2]
    dt = date.fromisoformat(f'{dt[:4]}-{dt[4:6]}-{dt[6:8]}')
    new_dt = dt + timedelta(days=1)
    desc = f'[{len(dataset.iloc[indexes])}/{len(dataset)}] {nm} {dt}'
    search_criteria = {
        "productType": product_type,
        "start": f'{dt}T00:00:00',
        # "end": f'{new_dt}T23:59:59',
        "end": f'{dt}T23:59:59',
        "geom": None,
        "items_per_page": 500,
    }

    for item in tqdm(dataset['geometry'].iloc[indexes], desc=desc, ascii=True):
        # poly = search_criteria["geom"] = Polygon(item).simplify(0.2, preserve_topology=False)
        poly = search_criteria["geom"] = Polygon(item)
        first_page, estimated = dag.search(**search_criteria)
        if estimated == 0:
            continue

        for elt in first_page:
            if {'1SDH', 'EW'} & set(elt.properties["title"].split('_')):
                continue
            if True:  # {'1SDV', 'IW'} & set(elt.properties["title"].split('_')):
                try:
                    product_path = elt.download(extract=False)
                    # cnt += 1
                except:
                    pass
    # print(f'got {cnt} products')

    # poly = search_criteria["geom"] = Polygon(unary_union(dataset['geometry'].iloc[indexes]))
    # page, estimated = dag.search(**search_criteria)
    # amt = len(scroll(page))
    # print(estimated)
    # if estimated > 500 and amt == 0:
    #     search_criteria["page"] = 1
    #     page, _ = dag.search(**search_criteria)
    #     amt += len(scroll(page))
    # if amt != 0:
    #     print(amt, end='')

print(f"skipped: {skip}")
