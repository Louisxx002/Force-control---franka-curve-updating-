"""PNP75 wire units confirmed by user on 2026-09-12: kgf and kgf*m.

Public sensor samples use SI, before tare/gravity compensation.
"""
STANDARD_GRAVITY = 9.80665
WRENCH_UNITS = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']
WIRE_UNITS = ['kgf', 'kgf', 'kgf', 'kgf*m', 'kgf*m', 'kgf*m']


def wire_to_si(values):
    return tuple(float(x) * STANDARD_GRAVITY for x in values)
