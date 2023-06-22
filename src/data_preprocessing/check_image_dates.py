from pathlib import Path

RAW_DIR = Path("data/raw/raw_processed/")
EXOLABS_DIR = Path("data/ExoLabs_classification_S2/")
CLOUDLESS_DIR = Path("data/S2cloudless/data/")
DYNAMICWORLD_DIR = Path("data/DynamicWorld/data/")


def check_raw_included(raw_dir: Path, other_dir: Path):
    for f in raw_dir.glob("*"):
        date = f.name[:8]
        pattern = f"*{date[:4]}?{date[4:6]}?{date[6:8]}*10m_SnowClass.tif"
        matching = list(other_dir.glob(pattern)) + list(other_dir.glob(f"*{date}*.tif"))
        if len(matching) != 1:
            print(f"{date} has {len(matching)} matches:", [m.name for m in matching])


def check_dw_included(dw_dir: Path, other_dir: Path):
    for f in dw_dir.glob("*"):
        date = f.name[13:21]
        pattern = f"*{date[:4]}?{date[4:6]}?{date[6:8]}*10m_SnowClass.tif"
        matching = list(other_dir.glob(pattern)) + list(other_dir.glob(f"*{date}*.tif"))
        if len(matching) != 1:
            print(f"{date} has {len(matching)} matches:", [m.name for m in matching])

def check_dw_raw(dw_dir: Path, raw_dir):
    mismatches = 0
    for f in dw_dir.glob("*"):
        date = f.name[13:21]
        matching = list(raw_dir.glob(f"*{date}*.tif"))
        if len(matching) != 1:
            print(f"{date} has {len(matching)} matches:", [m.name for m in matching])
            mismatches += 1
    print("Number of mismatches:", mismatches)

def check_raw_dw(raw_dir: Path, dw_dir: Path):
    mismatches = 0
    for f in raw_dir.glob("*"):
        date = f.name[:8]
        matching = list(dw_dir.glob(f"*{date}*.tif"))
        if len(matching) != 1:
            print(f"{date} has {len(matching)} matches:", [m.name for m in matching])
            mismatches += 1

    print("Number of mismatches:", mismatches)

def check_cloudless_exolabs(cloudless_dir: Path, exolabs_dir: Path):
    print("Images in s2cloudless:", len(list(cloudless_dir.glob("*"))))
    print("Images in Exolabs S2:", len(list(exolabs_dir.glob("*"))))
    for f in cloudless_dir.glob("*"):
        date = f.name[13:21]
        pattern = f"*{date[:4]}?{date[4:6]}?{date[6:8]}*10m_SnowClass.tif"
        matching = list(exolabs_dir.glob(pattern))
        if len(matching) != 1:
            print(f"{date} has {len(matching)} matches:", [m.name for m in matching])
            mismatches += 1

print("RAW -> EXOLABS")
check_raw_included(RAW_DIR, EXOLABS_DIR)
print("RAW -> CLOUDLESS")
check_raw_included(RAW_DIR, CLOUDLESS_DIR)

print("DW -> EXOLABS")
check_dw_included(DYNAMICWORLD_DIR, EXOLABS_DIR)
print("DW -> CLOUDLESS")
check_dw_included(DYNAMICWORLD_DIR, CLOUDLESS_DIR)

print("DW -> RAW")
check_dw_raw(DYNAMICWORLD_DIR, RAW_DIR)
print("RAW -> DW")
check_raw_dw(RAW_DIR, DYNAMICWORLD_DIR)

print("CLOUDLESS <-> EXOLABS")
check_cloudless_exolabs(CLOUDLESS_DIR, EXOLABS_DIR)