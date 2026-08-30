"""Adaptive-screening experiment numerical-mode setup."""

import os

import jax

if os.environ.get("ADAPTIVE_SCREENING_TRAINING") != "1":
    jax.config.update("jax_enable_x64", True)
