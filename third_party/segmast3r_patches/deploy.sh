#!/usr/bin/env bash
# Copy our untracked patches into the segmast3r submodule.
# Safe to re-run; refuses to clobber a newer file without --force.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$HERE/../segmast3r/src/models/mast3r_segfeat" && pwd)"
FORCE="${1:-}"

for f in double_softmax_matcher.py segment_attention.py; do
    src="$HERE/$f"; dst="$DEST/$f"
    if [[ -f "$dst" && "$dst" -nt "$src" && "$FORCE" != "--force" ]]; then
        echo "SKIP  $f (destination is newer; pass --force to overwrite)"
        continue
    fi
    cp -v "$src" "$dst"
done
echo "Done. Deployed into $DEST"
