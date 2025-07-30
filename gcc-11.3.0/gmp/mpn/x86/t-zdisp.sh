#! /bin/sh
#
# Copyright 2000 Free Software Foundation, Inc.
#
#  This file is part of the GNU MP Library.
#
#  The GNU MP Library is free software; you can redistribute it and/or modify
#  it under the terms of either:
#
#    * the GNU Lesser General Public License as published by the Free
#      Software Foundation; either version 3 of the License, or (at your
#      option) any later version.
#
#  or
#
#    * the GNU General Public License as published by the Free Software
#      Foundation; either version 2 of the License, or (at your option) any
#      later version.
#
#  or both in parallel, as here.
#
#  The GNU MP Library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received copies of the GNU General Public License and the
#  GNU Lesser General Public License along with the GNU MP Library.  If not,
#  see https://www.gnu.org/licenses/.


# Usage: cd $(builddir)/mpn
#        $(srcdir)/x86/t-zdisp.sh
#
# Run the Zdisp() macro instructions through the assembler to check
# the encodings used.  Mismatches are printed, no output means all ok.
#
# This program is only meant for use during development.  It can be
# run in the mpn build directory of any x86 configuration.
#
# For this test the assembler needs to generate byte sized 0
# displacements when given something like 0(%eax).  Recent versions of
# gas are suitable (eg. 2.9.x or 2.10.x).

set -e

cat >tmp-zdisptest.asm <<\EOF

include(`../config.m4')

dnl  Redefine Zdisp_match to output its pattern and encoding.
define(`Zdisp_match',
`define(`Zdisp_found',1)dnl
ifelse(`$2',0,`	$1	$2$3, $4')`'dnl
ifelse(`$3',0,`	$1	$2, $3$4')`'dnl

	.byte	$5
')
	.text
	Zdisp()
EOF

m4 tmp-zdisptest.asm >tmp-zdisptest.s
as -o tmp-zdisptest.o tmp-zdisptest.s

# Demand duplicates from the instruction patterns and byte encodings.
objdump -d tmp-zdisptest.o | awk '
/^ *[a-z0-9]+:/ {
	sub(/^ *[a-z0-9]+:/,"")
        print
}' | sort | uniq -u
