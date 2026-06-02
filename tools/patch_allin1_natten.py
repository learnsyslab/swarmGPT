"""Patch allin1's dinat.py for NATTEN >= 0.17 API compatibility.

allin1 1.1.0 imports natten1dav/natten1dqkrpb/natten2dav/natten2dqkrpb, which
were renamed in NATTEN 0.17 (new names: na1d_av, na1d_qk, etc.; rpb moved from
positional to keyword arg).  This script patches the installed file in-place.

Runs automatically via the music-patch pixi task (depends-on for analyze task).
Re-run manually after any `pixi install -e music` that upgrades allin1:
    pixi run -e music python tools/patch_allin1_natten.py
"""

import importlib.util
import sys
from pathlib import Path

OLD = "from natten.functional import natten1dav, natten1dqkrpb, natten2dav, natten2dqkrpb"
NEW = """\
from natten.functional import na1d_av as natten1dav, na2d_av as natten2dav
from natten.functional import na1d_qk as _na1d_qk, na2d_qk as _na2d_qk
def natten1dqkrpb(q, k, rpb, kernel_size, dilation): return _na1d_qk(q, k, kernel_size, dilation, rpb=rpb)
def natten2dqkrpb(q, k, rpb, kernel_size, dilation): return _na2d_qk(q, k, kernel_size, dilation, rpb=rpb)\
"""

spec = importlib.util.find_spec("allin1")
if spec is None:
    sys.exit("allin1 not found — are you in the music environment?")

dinat = Path(spec.origin).parent / "models" / "dinat.py"
text = dinat.read_text()

if OLD not in text:
    print(f"Already patched: {dinat}")
else:
    dinat.write_text(text.replace(OLD, NEW, 1))
    print(f"Patched: {dinat}")
