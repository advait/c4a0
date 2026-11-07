#!/bin/sh
# Real Pons with two planted faults: column 4 of 7265667621 and column 2 of a ply-30 position
# each score one higher than Pons said.
/tmp/c4a0-solver-oracle/c4solver "$@" | awk '
  $1 == "7265667621" { $5 = $5 + 1 }
  $1 == "643673432447521526115472111634" { $3 = $3 + 1 }
  { print; fflush() }'
