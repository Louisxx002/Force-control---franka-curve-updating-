import ast
from pathlib import Path

def test_execute_source_has_no_removed_pose_after_reference():
    source=Path('curve_wipe/execute.py').read_text()
    assert 'pose_after' not in source
    ast.parse(source)
