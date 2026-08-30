# segmast3r patches

These two modules are ours, but they live **inside** the `third_party/segmast3r`
submodule (`src/models/mast3r_segfeat/`), where `git status` reports them as
untracked. They are therefore versioned by nothing: a fresh clone, a submodule
re-checkout, or a transfer between machines loses them silently, and any model
that imports them then fails or -- worse -- silently resolves to something else.

They are kept here as the authoritative copy and deployed into the submodule.

| file | used by |
|---|---|
| `double_softmax_matcher.py` | SegMASt3R + LightGlue heads |
| `segment_attention.py`      | SegMASt3R segment cross-attention |

## Deploy

    bash third_party/segmast3r_patches/deploy.sh

Re-run after any `git submodule update` on `third_party/segmast3r`.
