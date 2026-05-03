from pathlib import Path


DATA_ROOT = Path("/mnt/hdd4tb/jaemo/data/LLP_original")
CACHE_ROOT = Path("/mnt/hdd4tb/jaemo/data/LLP/cached_avvp")
VOCAB_ROOT = Path("/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/vocabs")

TEST_CSV = DATA_ROOT / "AVVP_test_pd.csv"
EVAL_AUDIO_CSV = DATA_ROOT / "AVVP_eval_audio.csv"
EVAL_VISUAL_CSV = DATA_ROOT / "AVVP_eval_visual.csv"

DEFAULT_BACKBONE = "ClipClap"
DEFAULT_VOCAB = "v25"

LLP_CATS = [
    "Speech",
    "Car",
    "Cheering",
    "Dog",
    "Cat",
    "Frying_(food)",
    "Basketball_bounce",
    "Fire_alarm",
    "Chainsaw",
    "Cello",
    "Banjo",
    "Singing",
    "Chicken_rooster",
    "Violin_fiddle",
    "Vacuum_cleaner",
    "Baby_laughter",
    "Accordion",
    "Lawn_mower",
    "Motorcycle",
    "Helicopter",
    "Acoustic_guitar",
    "Telephone_bell_ringing",
    "Baby_cry_infant_cry",
    "Blender",
    "Clapping",
]

LLP_IDX = {label: idx for idx, label in enumerate(LLP_CATS)}
