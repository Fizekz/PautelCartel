"""One command: rebuild features, run the honest backtests, write the 2026 forecast."""
import fetch_sameparty
import national
import features
import backtest
import predict2026
import simulate
from common import WORK

if __name__ == "__main__":
    for t in fetch_sameparty.TITLES:   # cached in data/raw/wiki after the first run
        fetch_sameparty.fetch(t)
    national.main()
    h = features.build_history()
    h.to_csv(WORK / "features_hist.csv", index=False)
    f = features.build_2026()
    f.to_csv(WORK / "features_2026.csv", index=False)
    backtest.main()
    predict2026.main()
    simulate.main()
