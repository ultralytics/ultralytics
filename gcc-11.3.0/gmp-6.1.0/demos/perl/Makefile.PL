# Makefile for GMP perl module.

# Copyright 2001, 2003, 2004 Free Software Foundation, Inc.
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


# Bugs:
#
# When the generated Makefile re-runs "perl Makefile.PL" the GMP_BUILDDIR
# parameter is lost.


use ExtUtils::MakeMaker;


# Find and remove our parameters
@ARGV = map {
  if (/^GMP_BUILDDIR=(.*)/) {
    $GMP_BUILDDIR=$1; ();
  } else {
    $_;
  }
} (@ARGV);

$INC = "";
$LIBS = "-lgmp";
$OBJECT = "GMP.o";

if (defined $GMP_BUILDDIR) {
  if (! -f "$GMP_BUILDDIR/libgmp.la") {
    die "$GMP_BUILDDIR doesn't contain libgmp.la\n" .
	"if it's really a gmp build directory then go there and run \"make libgmp.la\"\n";
  }
  $INC = "-I$GMP_BUILDDIR $INC";
  $LIBS = "-L$GMP_BUILDDIR/.libs $LIBS";
}

WriteMakefile(
	      NAME         => 'GMP',
	      VERSION      => '2.00',
	      LIBS         => [$LIBS],
	      OBJECT       => $OBJECT,
	      INC          => $INC,
	      clean        => { FILES => 'test.tmp' },
	      PM => {
		'GMP.pm'      => '$(INST_LIBDIR)/GMP.pm',
		'GMP/Mpz.pm'  => '$(INST_LIBDIR)/GMP/Mpz.pm',
		'GMP/Mpq.pm'  => '$(INST_LIBDIR)/GMP/Mpq.pm',
		'GMP/Mpf.pm'  => '$(INST_LIBDIR)/GMP/Mpf.pm',
		'GMP/Rand.pm' => '$(INST_LIBDIR)/GMP/Rand.pm',
	      }
	      );


# Local variables:
# perl-indent-level: 2
# End:
