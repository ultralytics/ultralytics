#!/bin/sh
#
# A helper script for Makeasm.am .asm.lo rule.

# Copyright 2001 Free Software Foundation, Inc.
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


# Usage: m4-ccas --m4=M4 CC ... file.asm ...
#
# Process file.asm with the given M4 plus any -D arguments, then
# assemble with the given CC plus all arguments.
#
# The M4 command must be in a single --m4= argument, and will be split
# on whitespace.  When CC is invoked file.asm is replaced with a
# temporary .s file which is the M4 output.
#
# To allow parallel builds, the temp file name is based on the .asm
# file name, which will be the output object filename for all uses we
# put this script to.

M4=
CC=
DEFS=
ASM=
SEEN_O=no

for i in "$@"; do
  case $i in
    --m4=*)
      M4=`echo "$i" | sed 's/^--m4=//'`
      ;;
    -D*)
      DEFS="$DEFS $i"
      CC="$CC $i"
      ;;
    *.asm)
      if test -n "$ASM"; then
        echo "Only one .asm file permitted"
        exit 1
      fi
      BASENAME=`echo "$i" | sed -e 's/\.asm$//' -e 's/^.*[\\/:]//'`
      TMP=tmp-$BASENAME.s
      ASM=$i
      CC="$CC $TMP"
      ;;
    -o)
      SEEN_O=yes
      CC="$CC $i"
      ;;
    *)
      CC="$CC $i"
      ;;
  esac
done

if test -z "$M4"; then
  echo "No --m4 specified"
  exit 1
fi

if test -z "$ASM"; then
  echo "No .asm specified"
  exit 1
fi

# Libtool adds it's own -o when sending output to .libs/foo.o, but not
# when just wanting foo.o in the current directory.  We need an
# explicit -o in both cases since we're assembling tmp-foo.s.
#
if test $SEEN_O = no; then
  CC="$CC -o $BASENAME.o"
fi

echo "$M4 $DEFS $ASM >$TMP"
$M4 $DEFS $ASM >$TMP || exit

echo "$CC"
$CC || exit

# Comment this out to preserve .s intermediates
rm -f $TMP
