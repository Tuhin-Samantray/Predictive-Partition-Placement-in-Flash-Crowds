"""Seed derivation (G2): every random draw is seeded from master_seed in config."""
import hashlib


def md5_int(text):
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest(), "big")


def derive_seed(*labels):
    return md5_int(":".join(str(label) for label in labels))