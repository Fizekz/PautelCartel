"""pau-t: refetch the public Census inputs for the race graphs into pbase/data/raw_t/ (gitignored).
All files are US Census Bureau public-domain products on www2.census.gov (robots.txt has no rule for generic
agents). Generic User-Agent, no e-mail, 2 s between requests."""
import subprocess, sys, time
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_t"
UA = "Mozilla/5.0 (compatible; research-script)"
G = "https://www2.census.gov/geo/tiger/"
FILES = {
    # congressional-district cartographic boundaries (1:20m), one per set of lines
    "cd111.zip": G + "GENZ2010/gz_2010_us_500_11_20m.zip",
    "cd116.zip": G + "GENZ2018/shp/cb_2018_us_cd116_20m.zip",
    "cd118.zip": G + "GENZ2022/shp/cb_2022_us_cd118_20m.zip",
    "cd119.zip": G + "GENZ2024/shp/cb_2024_us_cd119_20m.zip",
    # county adjacency
    "county_adjacency2023.txt": "https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2023.txt",
    # metro / combined statistical area delineations (media-market proxy)
    "cbsa_list1_2023.xlsx": "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx",
    # ACS 2011-2015 county-to-county commuting flows
    "commuting_2015_table1.xlsx": "https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2015/commuting-flows-2015/table1.xlsx",
}


def main(names):
    RAW.mkdir(parents=True, exist_ok=True)
    for k, url in FILES.items():
        if names and k not in names:
            continue
        f = RAW / k
        if f.exists() and f.stat().st_size > 1000:
            print("have", k); continue
        r = subprocess.run(["curl", "-sfL", "-A", UA, "--max-time", "300", "-o", str(f), url])
        print("got" if r.returncode == 0 else f"FAIL ({r.returncode})", k, f.stat().st_size if f.exists() else 0)
        time.sleep(2)


if __name__ == "__main__":
    main(sys.argv[1:])
