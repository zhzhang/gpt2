#!/bin/bash

uv run torchrun --standalone --nproc_per_node=1 train.py
