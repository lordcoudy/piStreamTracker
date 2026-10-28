"""Keep the shared wide-angle correction finite and preserve other sensor tuning."""

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'tuning/imx219-160-colour.json'


def test_matching_camera_deployments_select_the_same_colour_profile():
    for name in ('config.yaml', 'config.pi4.yaml'):
        config = yaml.safe_load((ROOT / name).read_text())
        assert config['camera']['tuning_file'] == PROFILE
        assert (ROOT / PROFILE).is_file()


def test_colour_profile_only_adds_static_lens_shading_to_original_tuning():
    original = json.loads((ROOT / 'tuning/imx219-160-no-alsc.json').read_text())
    corrected = json.loads((ROOT / PROFILE).read_text())
    algorithms = corrected.pop('algorithms')
    unchanged = [item for item in algorithms if 'rpi.alsc' not in item]
    assert corrected == {key: value for key, value in original.items() if key != 'algorithms'}
    assert unchanged == original['algorithms']
    alsc = next(item['rpi.alsc'] for item in algorithms if 'rpi.alsc' in item)
    assert alsc['n_iter'] == 0  # Scene contents cannot adapt this measured correction.
    assert alsc['luminance_strength'] == 0  # Avoid extra gain/noise in dim corners.
    for key in ('calibrations_Cr', 'calibrations_Cb'):
        for entry in alsc[key]:
            assert entry['ct'] > 0
            assert len(entry['table']) == 16 * 12
            assert all(math.isfinite(value) and value > 0 for value in entry['table'])
