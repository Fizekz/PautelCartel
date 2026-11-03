"""pau-k runner. Usage: pk_run.py NAME [NAME ...]
NAME = '+'-joined component list ('base' = v6+nl), optional suffixes after ':' e.g. 'node+prim:clip3', ':s' (prim_dev_s),
':eco' (cells with the ecological turnout), ':plc<seed>' (joint placebo), ':ins' (add the in-sample scheme).
Results: results/pk/<NAME>.json; predictions work/stack/<NAME>_<scheme>.csv."""
import sys
import pk_stack as K


def parse(name):
    comps, _, opts = name.partition(":")
    comps = [] if comps == "base" else comps.split("+")
    kw, plc, schemes = {}, None, ["six", "pre18", "bt", "loyo8"]
    for o in filter(None, opts.split(",")):
        if o.startswith("clip"):
            kw["clip"] = float(o[4:])
        elif o == "s":
            kw["prim"] = "prim_dev_s"
        elif o == "eco":
            kw["cells_mode"] = "eco"
        elif o.startswith("plc"):
            plc = int(o[3:])
        elif o.startswith("w"):
            kw["pool_w"] = float(o[1:])
        elif o == "ins":
            schemes.append("ins")
        elif o == "sel":
            schemes = ["six", "pre18"]
    return comps, kw, plc, schemes


if __name__ == "__main__":
    for n in sys.argv[1:]:
        comps, kw, plc, schemes = parse(n)
        cfg = K.build(comps, plc=plc, **kw)
        h = K.placebo(plc, comps) if plc is not None else None
        try:
            K.run(n.replace(":", "__").replace(",", "_"), cfg, hist=h, schemes=schemes, save=plc is None)
        finally:
            K.restore()
