#!/usr/bin/python3
# -*- coding: utf8 -*-
# vim: set fileencoding=utf-8 :

from math import asin, pi


# Angle d'un demi-point
angle = asin(0.5/3.5)/(2*pi)

class Target:
    target_id = None
    sequence = []

    def __init__(self, target_id, sequence):
        self.target_id = target_id
        self.sequence = sequence

    def print(self):
        sequence_string = ""
        for val in self.sequence:
            if val:
                sequence_string += u"\u25CF"
            else:
                sequence_string += u"\u25CB"
        print(str(self.target_id) + " - " + sequence_string)

    def equals(self, sequence):
        if len(sequence) != len(self.sequence):
            return False
        elif sequence == self.sequence:
            return True
        else:
            altered_sequence = sequence.copy()
            for i in range(len(self.sequence)-1):
                last_value = altered_sequence.pop()
                altered_sequence.insert(0, last_value)
                if altered_sequence == self.sequence:
                    return True
            return False

    def getSignal(self):
        signal = []
        for i, point in enumerate(self.sequence):
            if point:
                # Position du centre du point
                center = i / len(self.sequence)
                if i == 0:
                    signal.append(1 - angle)
                    signal.append(angle)
                else:
                    signal.append(center - angle)
                    signal.append(center + angle)
        signal.sort()
        return "[" + ",".join(map("{0:1.5f}".format, signal)) + "]"

    def __repr__(self):
        return repr((self.target_id, self.sequence))