import json
import pandas as pd


def load_longmemeval_o_df():
    with open("longmemeval-cleaned/Long Meme Val Oracle.json", "r", encoding="utf-8") as f:
        longmemeval_o = json.load(f)
    return pd.DataFrame(longmemeval_o)


def load_longmemeval_s_df():
    with open("longmemeval-cleaned/Long Memeval Cleaned.json", "r", encoding="utf-8") as f:
        longmemeval_s = json.load(f)
    return pd.DataFrame(longmemeval_s)


def load_longmemeval_m_df():
    with open("longmemeval-cleaned/Long Meme Val Cleaned.json", "r", encoding="utf-8") as f:
        longmemeval_m = json.load(f)
    return pd.DataFrame(longmemeval_m)
