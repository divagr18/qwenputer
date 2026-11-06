#!/bin/sh
# Hash-checks fx's tracked sources (src/, build.zig, build.zig.zon) against commit 757bdb9:
# every blob must equal the commit's. Untracked files (a platform root copied in for a
# build) are listed separately; they are not fx's code.
#   check_src.sh FX_CHECKOUT
set -e
FX=$1
cd "$FX"
git ls-tree -r 757bdb9 -- src build.zig build.zig.zon | awk '{print $3" "$4}' > /tmp/fx_tree.txt
cut -d' ' -f2 /tmp/fx_tree.txt | git hash-object --stdin-paths > /tmp/fx_now.txt
n=$(wc -l < /tmp/fx_tree.txt)
bad=$(paste -d' ' /tmp/fx_tree.txt /tmp/fx_now.txt | awk '$1 != $3 {print $2}')
extra=$(comm -13 <(cut -d' ' -f2 /tmp/fx_tree.txt | grep '^src/' | sort) <(find src -type f | sort))
echo "fx files checked: $n; changed: $(echo -n "$bad" | grep -c . || true)"
[ -n "$bad" ] && echo "CHANGED: $bad" && exit 1
[ -n "$extra" ] && echo "extra (not fx's): $extra"
echo "fx sources identical to 757bdb9"
