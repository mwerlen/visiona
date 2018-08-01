#!/usr/bin/python3
# -*- coding: utf8 -*-
# vim: set fileencoding=utf-8 :

import logging
import os
from math import sin, cos, pi


log = logging.getLogger("latex_printer")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

TARGET_DIR = "./targets"

TEMPLATE = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
%\\usepackage[a0paper]{geometry}
\\usepackage[top=2cm, bottom=3cm, left=1cm, right=1cm]{geometry}
\\geometry{a4paper}


\\usepackage{fancyhdr}
\\pagestyle{fancy}
\\headheight=30pt
\\renewcommand{\\footrulewidth}{0.4pt}
\\cfoot{\\small Série 1}
\\chead{\\large Sogelink}
\\lhead{\\Huge@@}
\\rhead{\\Huge@@}
\\lfoot{\\Huge@@}
\\rfoot{\\Huge@@}

\\usepackage{tikz}
\\usepgflibrary{shapes}
%\\usetikzlibrary{shapes}
%\\usetikzlibrary{backgrounds,fit,shapes}
%\\usetikzlibrary{positioning}

\\pagenumbering{gobble}

\\begin{document}
\\begin{centering}
%\\begin{tikzpicture}
% Centers the image coordinate system to the center of the page
\\begin{tikzpicture}[remember picture,overlay,shift={(current page.center)}]

% Concentric circles
\\fill[black] (0,0) circle (8cm);  
\\fill[white] (0,0) circle (3.2cm);  
\\fill[black] (0,0) circle (0.1cm);  
\\node[below, black] at (0,-0.2) {\\Huge @@};

% Code
#---points-------#

\\end{tikzpicture}
\\end{centering}
\\end{document}
"""

def create_point(number, total):
    x = 5.6000 * cos(number / total * 2 * pi)
    y = 5.6000 * sin(number / total * 2 * pi) * -1 # oui on tourne dans le sens inverse
    return "\\fill[white] ( {0:1.5f},  {1:1.5f}) circle (0.8000cm);".format(x,y)

def create_points_content(target):
    points = ""
    for i,value in enumerate(target.sequence):
        if value:
            points += create_point(i,len(target.sequence))
            points += "\n"
    return points

def create_content(target):
    content = TEMPLATE.replace("@@", "{0:02d}".format(target.target_id))
    content = content.replace("#---points-------#", create_points_content(target))
    return content

def latex_print(target):
    target.print()
    file = TARGET_DIR + "/target_{0:02d}.tex".format(target.target_id)
    log.debug(file)
    target_latex = open(file, 'w')
    target_latex.write(create_content(target))
    target_latex.close()


def create_target_dir():
    directory = os.path.dirname(TARGET_DIR)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
