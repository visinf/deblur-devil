# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import os
import zipfile

from utils import system


def create_zip(filename,
               directory,
               include_extensions=('*.py', '*.txt', '*.md',
                                   '*.sh', '.c', '.cpp',
                                   '.cu', '.cuh', 'h')):
    filenames = []
    arcdir = os.path.basename(filename.split('.')[0])
    for ext in include_extensions:
        filenames += system.get_filenames(directory, match=ext)
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as file:
        for f in filenames:
            arcname = f.replace(directory, arcdir)
            file.write(f, arcname)
