#! /usr/bin/python

# Automatically detects the /dev/pcanusbN file for controlling the CAN device.
#
# Searches for all possible matches and uses regexp to look for a file of the
# form /dev/pcanusbN. If exactly one is found, prints it and returns it (so this
# is usable within another script). Otherwise, returns the empty string.
#
# Author: Felix Duvallet <felix.duvallet@epfl.ch>

import glob
import re


def pcan_search():
    # Look for all possible file matches.
    filelist = glob.glob("/dev/pcanusb*")

    # Search for the correct format: pcanusbN
    device_re = re.compile(".*pcanusb[0-9]+")

    gen = (x for x in filelist if device_re.search(x))
    valid_files = [x for x in gen]

    if len(valid_files) == 1:
        print(valid_files[0])
        return valid_files[0]
    else:
        return ""


if __name__ == "__main__":
    pcan_search()
