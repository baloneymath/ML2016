#!/usr/bin/env python3

import sys
from PIL import Image

fileName = sys.argv[1]
im = Image.open(fileName)
im.rotate(180).save("ans2.png")


