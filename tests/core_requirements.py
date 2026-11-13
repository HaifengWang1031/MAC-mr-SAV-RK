"""Check the one authoritative index of required core regression functions."""
from collections.abc import Iterable
import json
from pathlib import Path, PurePosixPath

MANIFEST = Path(__file__).with_suffix('.json')


def check_core_requirements(nodeids: Iterable[str], *, manifest: Path = MANIFEST) -> None:
    """Fail if an indexed function is malformed, absent or outside collected core."""
    groups = json.loads(manifest.read_text())
    if not isinstance(groups, dict) or not groups:
        raise ValueError('Core requirements must be a nonempty mechanism mapping')
    required: set[str] = set()
    for mechanism, nodes in groups.items():
        if not mechanism.strip() or not isinstance(nodes, list) or not nodes:
            raise ValueError('Every core mechanism needs regression functions')
        for node in nodes:
            if not isinstance(node, str):
                raise ValueError('Core regression identifiers must be strings')
            parts = node.split('::')
            path = PurePosixPath(parts[0])
            if (len(parts) != 2 or not parts[0].startswith('tests/core/') or
                    '..' in path.parts or path.suffix != '.py' or
                    not parts[-1].startswith('test_') or not parts[-1].isidentifier()):
                raise ValueError('Expected a core test function identifier: '+node)
            if node in required:
                raise ValueError('Duplicate core regression identifier: '+node)
            required.add(node)
    collected = {node.split('[', 1)[0] for node in nodeids}
    missing = sorted(required-collected)
    if missing:
        raise ValueError('Required core regressions not collected: '+', '.join(missing))
