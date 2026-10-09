"""Load filesystem locations from an explicitly selected external JSON config."""
from pathlib import Path
import os
import json

def configuration():
    value=os.environ.get('NSP_CONFIG_FILE')
    if not value:
        raise RuntimeError('Set NSP_CONFIG_FILE to your private JSON configuration file.')
    with Path(value).expanduser().open(encoding='utf-8-sig') as handle:
        return json.load(handle)

def resource(key):
    value=configuration()['resources'][key]
    if not isinstance(value,str) or not value:
        raise ValueError('Resource configuration values must be nonempty strings.')
    return value

def analysis_root():
    value=configuration().get('analysis_root')
    if not value:
        raise ValueError('Set analysis_root in your private configuration.')
    root=Path(value).expanduser().resolve()
    repository=Path(__file__).resolve().parents[1]
    if root==repository or repository in root.parents:
        raise ValueError('Use a private analysis workspace outside the public repository.')
    if not root.is_dir():raise FileNotFoundError('Private workspace does not exist.')
    return root
