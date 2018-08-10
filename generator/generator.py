#!/usr/bin/python3
# -*- coding: utf8 -*-
# vim: set fileencoding=utf-8 :

import logging;
from Target import *
import LaTeX_printer

log = logging.getLogger("generator")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

def generate_sequences(length):
    log.debug("Niveau : " + str(length))
    if length <= 1:
        yield [True]
        yield [False]
    else:
        for seq in generate_sequences(length-1):
            yield [True] + seq
            yield [False] + seq


def test_if_exists(sequence, targets):
    for target in targets:
        if target.equals(sequence):
            return True
    return False

def generate_targets(bits):
    targets = []
    count = 0
    for sequence in generate_sequences(bits):
        log.debug(str(sequence))
        if sequence.count(True) == 0:
            continue
        if test_if_exists(sequence, targets):
            continue
        else:
            count += 1
            target = Target(count, sequence)
            targets.append(target)
    return targets


if __name__ == "__main__":
    LaTeX_printer.create_target_dir()
    targets = generate_targets(8)
    sorted(targets, key=lambda target: target.sequence)
    log.info(str(len(targets))+" targets found")
    for target in targets:
        log.debug(target)
        # LaTeX_printer.latex_print(target)
        log.debug(target.getSignal())
    print(",\n".join(map(Target.getConfig, targets)))
