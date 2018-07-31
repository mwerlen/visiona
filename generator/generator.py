#!/usr/bin/python3
# -*- coding: utf8 -*-
# vim: set fileencoding=utf-8 :

import logging;
from Target import *

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
    targets = generate_targets(8)
    log.info(str(len(targets))+" targets found")
    log.info(targets[0].getSignal())