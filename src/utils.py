from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import List, Dict

def clean_feature_names(raw_names: List[str]) -> List[str]:
    names = []
    for n in raw_names:
        n = n.strip()
        n = n.replace("()", "")
        n = n.replace("(", "-").replace(")", "")
        n = n.replace(",", "-").replace(" ", "")
        n = n.replace("BodyBody", "Body")  # veri setindeki tekrar hatası
        names.append(n)
    return names

def build_feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    cols = np.array(columns)
    def pick(pattern: str) -> List[str]:
        rx = re.compile(pattern)
        return [c for c in cols if rx.search(c)]
    groups = {
        "time_body_acc_xyz": pick(r"^tBodyAcc-(X|Y|Z)$"),
        "time_gravity_acc_xyz": pick(r"^tGravityAcc-(X|Y|Z)$"),
        "time_body_gyro_xyz": pick(r"^tBodyGyro-(X|Y|Z)$"),
        "freq_body_acc_xyz": pick(r"^fBodyAcc-(X|Y|Z)$"),
        "freq_body_gyro_xyz": pick(r"^fBodyGyro-(X|Y|Z)$"),
        "acc_all": pick(r"Acc"),
        "gyro_all": pick(r"Gyro"),
        "time_all": pick(r"^t"),
        "freq_all": pick(r"^f")
    }
    return {k: v for k, v in groups.items() if len(v) > 0}
