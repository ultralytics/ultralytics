#!/usr/bin/perl -w

# Some sample GMP module operations

# Copyright 2001, 2004 Free Software Foundation, Inc.
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

use strict;


use GMP;
print "using GMP module $GMP::VERSION and GMP library ",GMP::version(),"\n";


use GMP::Mpz qw(:all);
print "the 200th fibonacci number is ", fib(200), "\n";
print "next prime after 10**30 is (probably) ", nextprime(mpz(10)**30), "\n";


use GMP::Mpq qw(:constants);
print "the 7th harmonic number is ", 1+1/2+1/3+1/4+1/5+1/6+1/7, "\n";
use GMP::Mpq qw(:noconstants);


use GMP::Mpf qw(mpf);
my $f = mpf(1,180);
$f >>= 180;
$f += 1;
print "a sample mpf is $f\n";
