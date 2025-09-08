import json
import os

import pandas as pd
import tqdm
from gracedb_sdk import Client
from pyarrow import feather
from requests import HTTPError


def gdb_sdk_fetch(start, end):
    client = Client(url="https://gracedb.ligo.org/api/")
    far_thresh = 1 / (6 * 60 * 60 * 24 * 30)  # 1/6 months
    events = client.events.search(
        query=f"CBC gpstime: {start} .. {end} far < {far_thresh} is_preferred_event: True"
    )
    events_jsons = [event for event in events]
    # event_ids = [event["graceid"] for event in events]
    # print(event_list) # event iterator like fetch all PE
    coinc = False
    if coinc:
        for event in events_jsons:
            event_id = event["graceid"]
            print(event_id)
            filename = "coinc.xml"
            # path = f'../PSD_study/O4_coinc/{start}_{end}/{event_id}'
            path = f"../PSD_study/O4_coinc/{start}_{end}/{event_id}"
            if not os.path.exists(path):
                os.makedirs(path)
            with open(f"{path}/{filename}", "wb") as f:
                read_file = client.events[event_id].files[filename].get()
                f.write(read_file.read())
                print(f"{path}/{filename}")

    return events_jsons


# def gdb_sdk_parallel(start, end, n_threads=2):
#     times = np.linspace(float(start), float(end), n_threads + 1)
#     events_jsons = Parallel(n_jobs=n_threads)(
#         delayed(gdb_sdk_fetch)(
#             times[n],
#             times[n + 1],
#         )
#         for n in range(n_threads)
#     )
#     # events = [sec for sec in events_jsons]


def fetched_data_to_df(fetched):
    to_pd = {}
    event_keys = [
        "created",
        "group",
        "graceid",
        "pipeline",
        "gpstime",
        "reporting_latency",
        "instruments",
        "search",
        "labels",
        "superevent",
    ]
    coinc_inspr_keys = [
        "snr",
        "combined_far",
        "mchirp",
    ]
    single_inspr_keys = [
        "mass1",
        "mass2",
        "spin1z",
        "spin2z",
    ]
    superevent_keys = ["gw_events", "preferred_event"]
    to_pd = {
        k: []
        for k in event_keys + coinc_inspr_keys + single_inspr_keys + superevent_keys
    }
    for event in tqdm.tqdm(fetched):
        for key in event_keys:
            to_pd[key].append(event[key])
        for coinc_key in coinc_inspr_keys:
            to_pd[coinc_key].append(
                event["extra_attributes"]["CoincInspiral"][coinc_key]
            )
        for single_key in single_inspr_keys:
            try:
                to_pd[single_key].append(
                    event["extra_attributes"]["SingleInspiral"][0][single_key]
                )
            except Exception(KeyError):
                print(event["graceid"])
                to_pd[single_key].append(None)
        for key in superevent_keys:
            try:
                S_ID = event["superevent"]
                to_pd[key].append(event["superevent_neighbours"][S_ID][key])
            except Exception(KeyError):
                to_pd[key].append(None)
    df = pd.DataFrame(to_pd)
    return df


def df_get_pastro(df):
    (
        df["pBNS"],
        df["pBBH"],
        df["pNSBH"],
        df["pTerr"],
        df["HasRemnant"],
        df["HasNS"],
        df["HasMassGap"],
    ) = None, None, None, None, None, None, None
    client = Client(url="https://gracedb.ligo.org/api/")  # , fail_if_noauth=True)
    for i, event in tqdm.tqdm(df.iterrows(), total=len(df)):
        # get p(astros)
        try:
            p_astro_dict = json.loads(
                client.events[event["graceid"]]
                .files[f"{event['pipeline'].lower()}.p_astro.json"]
                .get()
                .read()
            )
            (
                df.loc[i, "pBNS"],
                df.loc[i, "pBBH"],
                df.loc[i, "pNSBH"],
                df.loc[i, "pTerr"],
            ) = (
                p_astro_dict["BNS"],
                p_astro_dict["BBH"],
                p_astro_dict["NSBH"],
                p_astro_dict["Terrestrial"],
            )
        except Exception:
            pass

        # get em_brights
        em_bright_dict = None
        try:
            em_bright_filename = f"{event['pipeline'].lower()}.em_bright.json"
            em_bright_dict = json.loads(
                client.events[event["graceid"]].files[em_bright_filename].get().read()
            )
        except Exception(HTTPError):
            try:
                em_bright_filename = "em_bright.json"
                em_bright_dict = json.loads(
                    client.events[event["graceid"]]
                    .files[em_bright_filename]
                    .get()
                    .read()
                )
            except Exception:
                pass
        if em_bright_dict:  # UnboundLocalError: local variable 'em_bright_dict' referenced before assignment
            for key in ["HasRemnant", "HasNS", "HasMassGap"]:
                try:
                    df.loc[i, key] = em_bright_dict[key]
                except Exception(KeyError):
                    pass
    return df


if __name__ == "__main__":
    # MDC 10
    # offset = 94848000
    # start_time = 1357152000 # Jan 7, 2023
    # end_time = 1360608000 # Feb 16, 2023

    # MDC 11
    # offset = 98304000
    # start_time = 1360608000
    # end_time = 1364064000  # March 28, 2023

    # O4a
    # start_time = 1368994000
    # end_time = 1388812000

    # ER16
    # start_time = 1394896593
    # end_time = 1396710993

    # O3
    # start_time = 1238515300
    # end_time = 1239000000 #1268431100

    # O4b
    # start_time = 1396420100
    # end_time = 1398002003

    # O4 until now
    start_time = 1368994000
    end_time = 1441386345  # 1440947214 #lal_tconvert

    # 1441386345

    fetched = gdb_sdk_fetch(start_time, end_time)
    df = fetched_data_to_df(fetched)
    df = df_get_pastro(df)

    feather.write_feather(df, f"pref_events_df_{start_time}_{end_time}.fthr")
