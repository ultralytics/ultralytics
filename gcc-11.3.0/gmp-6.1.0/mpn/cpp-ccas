#!/bin/sh
#
# A helper script for Makeasm.am .S.lo rule.

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


# Usage: cpp-cc --cpp=CPP CC ... file.S ...
#
# Process file.S with the given CPP command plus any -D options in the
# rest of the arguments, then assemble with the given CC plus all
# arguments.
#
# The CPP command must be in a single --cpp= argument, and will be
# split on whitespace.  It should include -I options required.
#
# When CC is invoked, file.S is replaced with a temporary .s file
# which is the CPP output.
#
# Any lines starting with "#" are removed from the CPP output, usually
# these will be #line and #file markers from CPP, but they might also
# be comments from the .S.
#
# To allow parallel builds, the temp file name is based on the .S file
# name, which will be the output object filename for all uses we put
# this script to.

CPP=
CPPDEFS=
CC=
S=
SEEN_O=no

for i in "$@"; do
  case $i in
    --cpp=*)
      CPP=`echo "$i" | sed 's/^--cpp=//'`
      ;;
    -D*)
      CPPDEFS="$CPPDEFS $i"
      CC="$CC $i"
      ;;
    *.S)
      if test -n "$S"; then
        echo "Only one .S file permitted"
        exit 1
      fi
      BASENAME=`echo "$i" | sed -e 's/\.S$//' -e 's/^.*[\\/:]//'`
      S=$i
      TMP_I=tmp-$BASENAME.i
      TMP_S=tmp-$BASENAME.s
      CC="$CC $TMP_S"
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

if test -z "$CPP"; then
  echo "No --cpp specified"
  exit 1
fi

if test -z "$S"; then
  echo "No .S specified"
  exit 1
fi

# Libtool adds it's own -o when sending output to .libs/foo.o, but not
# when just wanting foo.o in the current directory.  We need an
# explicit -o in both cases since we're assembling tmp-foo.s.
#
if test $SEEN_O = no; then
  CC="$CC -o $BASENAME.o"
fi

echo "$CPP $CPPDEFS $S >$TMP_I"
$CPP $CPPDEFS $S >$TMP_I || exit

echo "grep -v '^#' $TMP_I >$TMP_S"
grep -v '^#' $TMP_I >$TMP_S

echo "$CC"
$CC || exit

# Comment this out to preserve .s intermediates
rm -f $TMP
