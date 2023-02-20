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

warnings.filterwarnings("ignore")
reg_path = 'E:/files/regions/2021'
workspace = 'E:/dag_img_2'

setup_logging(0)
# setup_logging(verbose=3, no_progress_bar=False)
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"

# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__USERNAME"] = ["t0pcup@yandex.ru", "kdmikhaylova_1@edu.hse.ru"][0]
# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__PASSWORD"] = ["jL7-iq4-GBM-RPe", "b8k-Jyy-NzS-jZ6"][0]

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
    lst, idx = ['W', 'I', 'S'], []
    my_ind = -1
    for obj in schema_dataset:
        my_ind += 1
        if obj['properties']['POLY_TYPE'] in lst:
            idx.append(my_ind)
    return idx


def scroll(pg):
    lst = []
    for elt in pg:
        # if {'1SDH', 'EW'} & set(elt.properties["title"].split('_')):
        #     continue
        if {'1SDV', 'IW'} & set(elt.properties["title"].split('_')):
            try:
                product_path = elt.download(extract=False)
                lst.append(product_path)
            except:
                pass
    return lst


for f in glob.glob(f'{reg_path}/*A_*.shp')[::-1]:
    # if 'SGRDREC_' in f:
    #     continue
    cnt, indexes = 0, verify(f)
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
        poly = search_criteria["geom"] = Polygon(item)
        first_page, estimated = dag.search(**search_criteria)
        # time.sleep(2)
        if estimated == 0:
            continue

        for elt in first_page:
            # if {'1SDH', 'EW'} & set(elt.properties["title"].split('_')):
            #     continue
            if {'1SDV', 'IW'} & set(elt.properties["title"].split('_')):
                try:
                    product_path = elt.download(extract=False)
                    cnt += 1
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
