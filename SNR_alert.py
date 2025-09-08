import requests
from astropy.table import Table
from astropy.time import Time


def parse_created_utc(created_str: str) -> Time:
    s = created_str.replace(" UTC", "")
    return Time(s, scale="utc", format="iso")


def get_network_snr(extra: dict):
    if not isinstance(extra, dict):
        return None
    return extra.get("network_snr") or extra.get("snr")


url = "https://gracedb.ligo.org/api/superevents/"
all_superevents = []
snr_data = []
created_data = []

while url:
    print(f"{url}\n")
    r = requests.get(url, headers={"Accept": "application/json"})
    data = r.json()
    all_superevents.extend(data["superevents"])

    for se in data.get("superevents", []):
        created = parse_created_utc(se["created"])
        extra = (
            se.get("preferred_event_data", {})
            .get("extra_attributes", {})
            .get("CoincInspiral", {})
        )
        snr = get_network_snr(extra)
        if snr is not None:
            snr_data.append((created, snr))

    url = data["links"].get("next")


tab = Table(
    rows=[(c.isot, s) for c, s in snr_data],  # convert Time to ISO string
    names=("created", "network_snr"),
)
# Save to CSV
tab.write("superevents_snr.csv", format="csv", overwrite=True)


for created, snr in snr_data:
    print(created, snr)
