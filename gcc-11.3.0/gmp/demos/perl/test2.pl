# GMP perl module tests (part 2)

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


# The following uses of :constants seem to provoke segvs in perl 5.005_03,
# so they're kept separate file to be run only on suitable perl versions.


use GMP::Mpz qw(:constants);
{
  my $a = 123;
  ok (UNIVERSAL::isa ($a, "GMP::Mpz"));
}
use GMP::Mpz qw(:noconstants);

use GMP::Mpq qw(:constants);
{
  my $a = 123;
  ok (UNIVERSAL::isa ($a, "GMP::Mpq"));
}
use GMP::Mpq qw(:noconstants);

use GMP::Mpf qw(:constants);
{
  my $a = 123;
  ok (UNIVERSAL::isa ($a, "GMP::Mpf"));
}
use GMP::Mpf qw(:noconstants);


# compiled constants unchanged by clrbit etc when re-executed
foreach (0, 1, 2) {
  use GMP::Mpz qw(:constants);
  my $a = 15;
  my $b = 6;
  use GMP::Mpz qw(:noconstants);
  clrbit ($a, 0);
  ok ($a == 14);
  setbit ($b, 0);
  ok ($b == 7);
}

1;


# Local variables:
# perl-indent-level: 2
# End:
