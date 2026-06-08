#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import torch
import numpy as np
import pandas as pd

from pathlib import Path

from lightglue import SuperPoint
from lightglue import LightGlue
from lightglue.utils import rbd

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if device == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# In[2]:


def load_resized_image(path):

    img = cv2.imread(str(path))

    img = cv2.resize(
        img,
        (320, 180)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    img = (
        torch.from_numpy(img)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )

    return img.to(device)
    


# In[3]:


DATASET_ROOT = Path(
    r"E:\Nordland Dataset\nordland-part-2020\nordland-part-2020"
)

SUMMER_DIR = DATASET_ROOT / "summer"
WINTER_DIR = DATASET_ROOT / "winter"

summer_images = sorted(
    list(SUMMER_DIR.glob("*.png"))
)

winter_images = sorted(
    list(WINTER_DIR.glob("*.png"))
)

print(len(summer_images))
print(len(winter_images))


# In[4]:


extractor = (
    SuperPoint(
        max_num_keypoints=512
    )
    .eval()
    #.half()
    .to(device)
)

matcher = (
    LightGlue(
        features="superpoint"
    )
    .eval()
    .to(device)
)

print("Models Loaded")


# In[5]:


def profile_pipeline(
    summer_img,
    winter_img
):

    timings = {}

    # -------------------
    # Data Loading
    # -------------------

    start = time.perf_counter()

    image0 = load_resized_image(
        summer_img
    )

    image1 = load_resized_image(
        winter_img
    )

    if device == "cuda":
        torch.cuda.synchronize()

    timings["loading_ms"] = (
        time.perf_counter() - start
    ) * 1000

    # -------------------
    # Feature Extraction
    # -------------------

    start = time.perf_counter()

    with torch.no_grad():

        feats0 = extractor.extract(
            image0
        )

        feats1 = extractor.extract(
            image1
        )

    if device == "cuda":
        torch.cuda.synchronize()

    timings["feature_ms"] = (
        time.perf_counter() - start
    ) * 1000

    # -------------------
    # Matching
    # -------------------

    start = time.perf_counter()

    with torch.no_grad():

        matches01 = matcher({
            "image0": feats0,
            "image1": feats1
        })

    if device == "cuda":
        torch.cuda.synchronize()

    timings["matching_ms"] = (
        time.perf_counter() - start
    ) * 1000

    feats0, feats1, matches01 = [
        rbd(x)
        for x in [
            feats0,
            feats1,
            matches01
        ]
    ]

    matches = matches01["matches"]

    points0 = (
        feats0["keypoints"]
        [matches[:, 0]]
        .cpu()
        .numpy()
    )

    points1 = (
        feats1["keypoints"]
        [matches[:, 1]]
        .cpu()
        .numpy()
    )

    # -------------------
    # Homography
    # -------------------

    start = time.perf_counter()

    if len(points0) >= 4:

        H, mask = cv2.findHomography(
            points0,
            points1,
            cv2.USAC_MAGSAC,
            2.0
        )

    else:

        H = None
        mask = None

    timings["homography_ms"] = (
        time.perf_counter() - start
    ) * 1000

    timings["total_ms"] = (
        timings["loading_ms"]
        +
        timings["feature_ms"]
        +
        timings["matching_ms"]
        +
        timings["homography_ms"]
    )

    return timings


# In[6]:


torch.cuda.empty_cache()

for _ in range(5):

    profile_pipeline(
        summer_images[250],
        winter_images[250]
    )

print("Warmup Complete")


# In[7]:


test_frames = [
    250,
    625,
    1600,
    1625
]

latencies = []

for idx in test_frames:

    t = profile_pipeline(
        summer_images[idx],
        winter_images[idx]
    )

    latencies.append(
        t["total_ms"]
    )


# In[8]:


import numpy as np

print(
    "Average:",
    round(np.mean(latencies), 2),
    "ms"
)

print(
    "Minimum:",
    round(np.min(latencies), 2),
    "ms"
)

print(
    "Maximum:",
    round(np.max(latencies), 2),
    "ms"
)


# In[9]:


selected_frames = [
    250,
    625,
    1600,
    1625
]


# In[10]:


results = []

for idx in selected_frames:

    t = profile_pipeline(
        summer_images[idx],
        winter_images[idx]
    )

    results.append([
        idx,
        round(t["loading_ms"],2),
        round(t["feature_ms"],2),
        round(t["matching_ms"],2),
        round(t["homography_ms"],2),
        round(t["total_ms"],2)
    ])

    print(
        f"Frame {idx} done"
    )


# In[11]:


df = pd.DataFrame(
    results,
    columns=[
        "Frame",
        "Loading(ms)",
        "Feature(ms)",
        "Matching(ms)",
        "Homography(ms)",
        "Total(ms)"
    ]
)

df


# In[12]:


avg = df.mean(
    numeric_only=True
)

avg


# In[13]:


OUTPUT_DIR = Path(
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

df.to_csv(
    OUTPUT_DIR /
    "latency_report.csv",
    index=False
)

print(
    "Latency Report Saved"
)


# In[14]:


print(
    "Average Total Latency:",
    round(
        avg["Total(ms)"],
        2
    ),
    "ms"
)


# In[15]:


print("Device:", device)

if device == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA Available:",
        torch.cuda.is_available()
    )

    print(
        "FP16 Enabled:",
        True
    )





