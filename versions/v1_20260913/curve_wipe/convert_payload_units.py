"""Offline conversion of legacy PNP75 calibration samples; preserves source."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .units import STANDARD_GRAVITY, WRENCH_UNITS, WIRE_UNITS, wire_to_si


def convert(data):
    if data.get('wrench_units') is not None or data.get('schema_version') != 1:
        raise ValueError('Only untagged legacy schema 1 samples may be converted once')
    result = copy.deepcopy(data)
    for sample in result['train'] + result.get('holdout', []):
        for key in ('raw_mean_6', 'raw_std_6'):
            sample[key] = list(wire_to_si(sample[key]))
        sample['raw_samples'] = [row[:1] + list(wire_to_si(row[1:]))
                                 for row in sample['raw_samples']]
    result.update(schema_version=2, wrench_units=WRENCH_UNITS,
                  original_wire_units=WIRE_UNITS,
                  wire_to_si_factor=STANDARD_GRAVITY,
                  unit_source='User confirmed current PNP75 kg/kgm outputs, 2026-09-12')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    raw = a.source.read_bytes()
    result = convert(json.loads(raw))
    result.update(original_source=str(a.source.resolve()),
                  original_source_sha256=hashlib.sha256(raw).hexdigest())
    with a.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(a.output)


if __name__ == '__main__':
    main()
