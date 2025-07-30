#!/usr/bin/perl -w

# GMP perl module tests

# Copyright 2001-2003 Free Software Foundation, Inc.
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


# These tests aim to exercise the many possible combinations of operands
# etc, and to run all functions at least once, which if nothing else will
# check everything intended is in the :all list.
#
# Use the following in .emacs to match test failure messages.
#
# ;; perl "Test" module error messages
# (eval-after-load "compile"
#   '(add-to-list
#     'compilation-error-regexp-alist
#     '("^.*Failed test [0-9]+ in \\([^ ]+\\) at line \\([0-9]+\\)" 1 2)))


use strict;
use Test;

BEGIN {
  plan tests => 123,
  onfail => sub { print "there were failures\n" },
}

use GMP qw(:all);
use GMP::Mpz qw(:all);
use GMP::Mpq qw(:all);
use GMP::Mpf qw(:all);
use GMP::Rand qw(:all);

use GMP::Mpz qw(:constants);
use GMP::Mpz qw(:noconstants);
use GMP::Mpq qw(:constants);
use GMP::Mpq qw(:noconstants);
use GMP::Mpf qw(:constants);
use GMP::Mpf qw(:noconstants);

package Mytie;
use Exporter;
use vars  qw($val $fetched $stored);
$val = 0;
$fetched = 0;
$stored = 0;
sub TIESCALAR {
  my ($class, $newval) = @_;
  my $var = 'mytie dummy refed var';
  $val = $newval;
  $fetched = 0;
  $stored = 0;
  return bless \$var, $class;
}
sub FETCH {
  my ($self) = @_;
  $fetched++;
  return $val;
}
sub STORE {
  my ($self, $newval) = @_;
  $val = $newval;
  $stored++;
}
package main;

# check Mytie does what it should
{ tie my $t, 'Mytie', 123;
  ok ($Mytie::val == 123);
  $Mytie::val = 456;
  ok ($t == 456);
  $t = 789;
  ok ($Mytie::val == 789);
}


# Usage: str(x)
# Return x forced to a string, not a PVIV.
#
sub str {
  my $s = "$_[0]" . "";
  return $s;
}

my $ivnv_2p128 = 65536.0 * 65536.0 * 65536.0 * 65536.0
               * 65536.0 * 65536.0 * 65536.0 * 65536.0;
kill (0, $ivnv_2p128);
my $str_2p128 = '340282366920938463463374607431768211456';

my $uv_max = ~ 0;
my $uv_max_str = ~ 0;
$uv_max_str = "$uv_max_str";
$uv_max_str = "" . "$uv_max_str";


#------------------------------------------------------------------------------
# GMP::version

use GMP qw(version);
print '$GMP::VERSION ',$GMP::VERSION,' GMP::version() ',version(),"\n";


#------------------------------------------------------------------------------
# GMP::Mpz::new

ok (mpz(0) == 0);
ok (mpz('0') == 0);
ok (mpz(substr('101',1,1)) == 0);
ok (mpz(0.0) == 0);
ok (mpz(mpz(0)) == 0);
ok (mpz(mpq(0)) == 0);
ok (mpz(mpf(0)) == 0);

{ tie my $t, 'Mytie', 0;
  ok (mpz($t) == 0);
  ok ($Mytie::fetched > 0);
}
{ tie my $t, 'Mytie', '0';
  ok (mpz($t) == 0);
  ok ($Mytie::fetched > 0);
}
{ tie my $t, 'Mytie', substr('101',1,1); ok (mpz($t) == 0); }
{ tie my $t, 'Mytie', 0.0; ok (mpz($t) == 0); }
{ tie my $t, 'Mytie', mpz(0); ok (mpz($t) == 0); }
{ tie my $t, 'Mytie', mpq(0); ok (mpz($t) == 0); }
{ tie my $t, 'Mytie', mpf(0); ok (mpz($t) == 0); }

ok (mpz(-123) == -123);
ok (mpz('-123') == -123);
ok (mpz(substr('1-1231',1,4)) == -123);
ok (mpz(-123.0) == -123);
ok (mpz(mpz(-123)) == -123);
ok (mpz(mpq(-123)) == -123);
ok (mpz(mpf(-123)) == -123);

{ tie my $t, 'Mytie', -123; ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', '-123'; ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', substr('1-1231',1,4); ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', -123.0; ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', mpz(-123); ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', mpq(-123); ok (mpz($t) == -123); }
{ tie my $t, 'Mytie', mpf(-123); ok (mpz($t) == -123); }

ok (mpz($ivnv_2p128) == $str_2p128);
{ tie my $t, 'Mytie', $ivnv_2p128; ok (mpz($t) == $str_2p128); }

ok (mpz($uv_max) > 0);
ok (mpz($uv_max) == mpz($uv_max_str));
{ tie my $t, 'Mytie', $uv_max; ok (mpz($t) > 0); }
{ tie my $t, 'Mytie', $uv_max; ok (mpz($t) == mpz($uv_max_str)); }

{ my $s = '999999999999999999999999999999';
  kill (0, $s);
  ok (mpz($s) == '999999999999999999999999999999');
  tie my $t, 'Mytie', $s;
  ok (mpz($t) == '999999999999999999999999999999');
}

#------------------------------------------------------------------------------
# GMP::Mpz::overload_abs

ok (abs(mpz(0)) == 0);
ok (abs(mpz(123)) == 123);
ok (abs(mpz(-123)) == 123);

{ my $x = mpz(-123); $x = abs($x); ok ($x == 123); }
{ my $x = mpz(0);    $x = abs($x); ok ($x == 0);   }
{ my $x = mpz(123);  $x = abs($x); ok ($x == 123); }

{ tie my $t, 'Mytie', mpz(0); ok (abs($t) == 0); }
{ tie my $t, 'Mytie', mpz(123); ok (abs($t) == 123); }
{ tie my $t, 'Mytie', mpz(-123); ok (abs($t) == 123); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_add

ok (mpz(0) + 1 == 1);
ok (mpz(-1) + 1 == 0);
ok (1 + mpz(0) == 1);
ok (1 + mpz(-1) == 0);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_addeq

{ my $a = mpz(7); $a += 1; ok ($a == 8); }
{ my $a = mpz(7); my $b = $a; $a += 1; ok ($a == 8); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_and

ok ((mpz(3) & 1) == 1);
ok ((mpz(3) & 4) == 0);

{ my $a = mpz(3); $a &= 1; ok ($a == 1); }
{ my $a = mpz(3); $a &= 4; ok ($a == 0); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_bool

if (mpz(0))   { ok (0); } else { ok (1); }
if (mpz(123)) { ok (1); } else { ok (0); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_com

ok (~ mpz(0) == -1);
ok (~ mpz(1) == -2);
ok (~ mpz(-2) == 1);
ok (~ mpz(0xFF) == -0x100);
ok (~ mpz(-0x100) == 0xFF);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_dec

{ my $a = mpz(0); ok ($a-- == 0); ok ($a == -1); }
{ my $a = mpz(0); ok (--$a == -1); }

{ my $a = mpz(0); my $b = $a; $a--; ok ($a == -1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_div

ok (mpz(6) / 2 == 3);
ok (mpz(-6) / 2 == -3);
ok (mpz(6) / -2 == -3);
ok (mpz(-6) / -2 == 3);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_diveq

{ my $a = mpz(21); $a /= 3; ok ($a == 7); }
{ my $a = mpz(21); my $b = $a; $a /= 3; ok ($a == 7); ok ($b == 21); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_eq

{ my $a = mpz(0);
  my $b = $a;
  $a = mpz(1);
  ok ($a == 1);
  ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_inc

{ my $a = mpz(0); ok ($a++ == 0); ok ($a == 1); }
{ my $a = mpz(0); ok (++$a == 1); }

{ my $a = mpz(0); my $b = $a; $a++; ok ($a == 1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_ior

ok ((mpz(3) | 1) == 3);
ok ((mpz(3) | 4) == 7);

{ my $a = mpz(3); $a |= 1; ok ($a == 3); }
{ my $a = mpz(3); $a |= 4; ok ($a == 7); }

ok ((mpz("0xAA") | mpz("0x55")) == mpz("0xFF"));

#------------------------------------------------------------------------------
# GMP::Mpz::overload_lshift

{ my $a = mpz(7) << 1; ok ($a == 14); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_lshifteq

{ my $a = mpz(7); $a <<= 1; ok ($a == 14); }
{ my $a = mpz(7); my $b = $a; $a <<= 1; ok ($a == 14); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_mul

ok (mpz(2) * 3 == 6);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_muleq

{ my $a = mpz(7); $a *= 3;  ok ($a == 21); }
{ my $a = mpz(7); my $b = $a; $a *= 3;  ok ($a == 21); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_neg

ok (- mpz(0) == 0);
ok (- mpz(123) == -123);
ok (- mpz(-123) == 123);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_not

if (not mpz(0))   { ok (1); } else { ok (0); }
if (not mpz(123)) { ok (0); } else { ok (1); }

ok ((! mpz(0)) == 1);
ok ((! mpz(123)) == 0);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_pow

ok (mpz(0) ** 1 == 0);
ok (mpz(1) ** 1 == 1);
ok (mpz(2) ** 0 == 1);
ok (mpz(2) ** 1 == 2);
ok (mpz(2) ** 2 == 4);
ok (mpz(2) ** 3 == 8);
ok (mpz(2) ** 4 == 16);

ok (mpz(0) ** mpz(1) == 0);
ok (mpz(1) ** mpz(1) == 1);
ok (mpz(2) ** mpz(0) == 1);
ok (mpz(2) ** mpz(1) == 2);
ok (mpz(2) ** mpz(2) == 4);
ok (mpz(2) ** mpz(3) == 8);
ok (mpz(2) ** mpz(4) == 16);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_poweq

{ my $a = mpz(3); $a **= 4; ok ($a == 81); }
{ my $a = mpz(3); my $b = $a; $a **= 4; ok ($a == 81); ok ($b == 3); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_rem

ok (mpz(-8) % 3 == -2);
ok (mpz(-7) % 3 == -1);
ok (mpz(-6) % 3 == 0);
ok (mpz(6) % 3 == 0);
ok (mpz(7) % 3 == 1);
ok (mpz(8) % 3 == 2);

{ my $a = mpz(24); $a %= 7; ok ($a == 3); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_rshift

{ my $a = mpz(32) >> 1; ok ($a == 16); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_rshifteq

{ my $a = mpz(32); $a >>= 1; ok ($a == 16); }
{ my $a = mpz(32); my $b = $a; $a >>= 1; ok ($a == 16); ok ($b == 32); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_spaceship

ok (mpz(0) < 1);
ok (mpz(0) > -1);

ok (mpz(0) != 1);
ok (mpz(0) != -1);
ok (mpz(1) != 0);
ok (mpz(1) != -1);
ok (mpz(-1) != 0);
ok (mpz(-1) != 1);

ok (mpz(0) < 1.0);
ok (mpz(0) < '1');
ok (mpz(0) < substr('-1',1,1));
ok (mpz(0) < mpz(1));
ok (mpz(0) < mpq(1));
ok (mpz(0) < mpf(1));
ok (mpz(0) < $uv_max);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_sqrt

ok (sqrt(mpz(0)) == 0);
ok (sqrt(mpz(1)) == 1);
ok (sqrt(mpz(4)) == 2);
ok (sqrt(mpz(81)) == 9);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_string

{ my $x = mpz(0);    ok("$x" eq "0"); }
{ my $x = mpz(123);  ok("$x" eq "123"); }
{ my $x = mpz(-123); ok("$x" eq "-123"); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_sub

ok (mpz(0) - 1 == -1);
ok (mpz(1) - 1 == 0);
ok (1 - mpz(0) == 1);
ok (1 - mpz(1) == 0);

#------------------------------------------------------------------------------
# GMP::Mpz::overload_subeq

{ my $a = mpz(7); $a -= 1; ok ($a == 6); }
{ my $a = mpz(7); my $b = $a; $a -= 1; ok ($a == 6); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpz::overload_xor

ok ((mpz(3) ^ 1) == 2);
ok ((mpz(3) ^ 4) == 7);

{ my $a = mpz(3); $a ^= 1; ok ($a == 2); }
{ my $a = mpz(3); $a ^= 4; ok ($a == 7); }


#------------------------------------------------------------------------------
# GMP::Mpz::bin

ok (bin(2,0) == 1);
ok (bin(2,1) == 2);
ok (bin(2,2) == 1);

ok (bin(3,0) == 1);
ok (bin(3,1) == 3);
ok (bin(3,2) == 3);
ok (bin(3,3) == 1);


#------------------------------------------------------------------------------
# GMP::Mpz::cdiv

{ my ($q, $r);
  ($q, $r) = cdiv (16, 3);
  ok ($q == 6);
  ok ($r == -2);
  ($q, $r) = cdiv (16, -3);
  ok ($q == -5);
  ok ($r == 1);
  ($q, $r) = cdiv (-16, 3);
  ok ($q == -5);
  ok ($r == -1);
  ($q, $r) = cdiv (-16, -3);
  ok ($q == 6);
  ok ($r == 2);
}


#------------------------------------------------------------------------------
# GMP::Mpz::cdiv_2exp

{ my ($q, $r);
  ($q, $r) = cdiv_2exp (23, 2);
  ok ($q == 6);
  ok ($r == -1);
  ($q, $r) = cdiv_2exp (-23, 2);
  ok ($q == -5);
  ok ($r == -3);
}


#------------------------------------------------------------------------------
# GMP::Mpz::clrbit

{ my $a = mpz(3); clrbit ($a, 1); ok ($a == 1);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }
{ my $a = mpz(3); clrbit ($a, 2); ok ($a == 3);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }

{ my $a = 3; clrbit ($a, 1); ok ($a == 1);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }
{ my $a = 3; clrbit ($a, 2); ok ($a == 3);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }

# mutate only given variable
{ my $a = mpz(3);
  my $b = $a;
  clrbit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
}
{ my $a = 3;
  my $b = $a;
  clrbit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
}

{ tie my $a, 'Mytie', mpz(3);
  clrbit ($a, 1);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 1);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}
{ tie my $a, 'Mytie', 3;
  clrbit ($a, 1);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 1);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}

{ my $b = mpz(3);
  tie my $a, 'Mytie', $b;
  clrbit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
  ok (tied($a));
}
{ my $b = 3;
  tie my $a, 'Mytie', $b;
  clrbit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
  ok (tied($a));
}

#------------------------------------------------------------------------------
# GMP::Mpz::combit

{ my $a = mpz(3); combit ($a, 1); ok ($a == 1);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }
{ my $a = mpz(3); combit ($a, 2); ok ($a == 7);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }

{ my $a = 3; combit ($a, 1); ok ($a == 1);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }
{ my $a = 3; combit ($a, 2); ok ($a == 7);
  ok (UNIVERSAL::isa($a,"GMP::Mpz")); }

# mutate only given variable
{ my $a = mpz(3);
  my $b = $a;
  combit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
}
{ my $a = 3;
  my $b = $a;
  combit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
}

{ tie my $a, 'Mytie', mpz(3);
  combit ($a, 2);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 7);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}
{ tie my $a, 'Mytie', 3;
  combit ($a, 2);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 7);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}

{ my $b = mpz(3);
  tie my $a, 'Mytie', $b;
  combit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
  ok (tied($a));
}
{ my $b = 3;
  tie my $a, 'Mytie', $b;
  combit ($a, 0);
  ok ($a == 2);
  ok ($b == 3);
  ok (tied($a));
}

#------------------------------------------------------------------------------
# GMP::Mpz::congruent_p

ok (  congruent_p (21, 0, 7));
ok (! congruent_p (21, 1, 7));
ok (  congruent_p (21, 5, 8));
ok (! congruent_p (21, 6, 8));


#------------------------------------------------------------------------------
# GMP::Mpz::congruent_2exp_p

ok (  congruent_2exp_p (20, 0, 2));
ok (! congruent_2exp_p (21, 0, 2));
ok (! congruent_2exp_p (20, 1, 2));

#------------------------------------------------------------------------------
# GMP::Mpz::divexact

ok (divexact(27,3) == 9);
ok (divexact(27,-3) == -9);
ok (divexact(-27,3) == -9);
ok (divexact(-27,-3) == 9);

#------------------------------------------------------------------------------
# GMP::Mpz::divisible_p

ok (  divisible_p (21, 7));
ok (! divisible_p (21, 8));

#------------------------------------------------------------------------------
# GMP::Mpz::divisible_2exp_p

ok (  divisible_2exp_p (20, 2));
ok (! divisible_2exp_p (21, 2));

#------------------------------------------------------------------------------
# GMP::Mpz::even_p

ok (! even_p(mpz(-3)));
ok (  even_p(mpz(-2)));
ok (! even_p(mpz(-1)));
ok (  even_p(mpz(0)));
ok (! even_p(mpz(1)));
ok (  even_p(mpz(2)));
ok (! even_p(mpz(3)));

#------------------------------------------------------------------------------
# GMP::Mpz::export

{ my $s = mpz_export (1, 2, 1, 0, "0x61626364");
  ok ($s eq 'abcd'); }
{ my $s = mpz_export (-1, 2, 1, 0, "0x61626364");
  ok ($s eq 'cdab'); }
{ my $s = mpz_export (1, 2, -1, 0, "0x61626364");
  ok ($s eq 'badc'); }
{ my $s = mpz_export (-1, 2, -1, 0, "0x61626364");
  ok ($s eq 'dcba'); }

#------------------------------------------------------------------------------
# GMP::Mpz::fac

ok (fac(0) == 1);
ok (fac(1) == 1);
ok (fac(2) == 2);
ok (fac(3) == 6);
ok (fac(4) == 24);
ok (fac(5) == 120);

#------------------------------------------------------------------------------
# GMP::Mpz::fdiv

{ my ($q, $r);
  ($q, $r) = fdiv (16, 3);
  ok ($q == 5);
  ok ($r == 1);
  ($q, $r) = fdiv (16, -3);
  ok ($q == -6);
  ok ($r == -2);
  ($q, $r) = fdiv (-16, 3);
  ok ($q == -6);
  ok ($r == 2);
  ($q, $r) = fdiv (-16, -3);
  ok ($q == 5);
  ok ($r == -1);
}

#------------------------------------------------------------------------------
# GMP::Mpz::fdiv_2exp

{ my ($q, $r);
  ($q, $r) = fdiv_2exp (23, 2);
  ok ($q == 5);
  ok ($r == 3);
  ($q, $r) = fdiv_2exp (-23, 2);
  ok ($q == -6);
  ok ($r == 1);
}

#------------------------------------------------------------------------------
# GMP::Mpz::fib

ok (fib(0) == 0);
ok (fib(1) == 1);
ok (fib(2) == 1);
ok (fib(3) == 2);
ok (fib(4) == 3);
ok (fib(5) == 5);
ok (fib(6) == 8);

#------------------------------------------------------------------------------
# GMP::Mpz::fib2

{ my ($a, $b) = fib2(0); ok($a==0); ok($b==1); }
{ my ($a, $b) = fib2(1); ok($a==1); ok($b==0); }
{ my ($a, $b) = fib2(2); ok($a==1); ok($b==1); }
{ my ($a, $b) = fib2(3); ok($a==2); ok($b==1); }
{ my ($a, $b) = fib2(4); ok($a==3); ok($b==2); }
{ my ($a, $b) = fib2(5); ok($a==5); ok($b==3); }
{ my ($a, $b) = fib2(6); ok($a==8); ok($b==5); }

#------------------------------------------------------------------------------
# GMP::Mpz::gcd

ok (gcd (21) == 21);
ok (gcd (21,15) == 3);
ok (gcd (21,15,30,57) == 3);
ok (gcd (21,-15) == 3);
ok (gcd (-21,15) == 3);
ok (gcd (-21,-15) == 3);

#------------------------------------------------------------------------------
# GMP::Mpz::gcdext

{
  my ($g, $x, $y) = gcdext (3,5);
  ok ($g == 1);
  ok ($x == 2);
  ok ($y == -1);
}

#------------------------------------------------------------------------------
# GMP::Mpz::hamdist

ok (hamdist(5,7) == 1);

#------------------------------------------------------------------------------
# GMP::Mpz::import

{ my $z = mpz_import (1, 2, 1, 0, 'abcd');
  ok ($z == 0x61626364); }
{ my $z = mpz_import (-1, 2, 1, 0, 'abcd');
  ok ($z == 0x63646162); }
{ my $z = mpz_import (1, 2, -1, 0, 'abcd');
  ok ($z == 0x62616463); }
{ my $z = mpz_import (-1, 2, -1, 0, 'abcd');
  ok ($z == 0x64636261); }

#------------------------------------------------------------------------------
# GMP::Mpz::invert

ok (invert(1,123) == 1);
ok (invert(6,7) == 6);
ok (! defined invert(2,8));

#------------------------------------------------------------------------------
# GMP::Mpz::jacobi, GMP::Mpz::kronecker

foreach my $i ([  1, 19,  1 ],
	       [  4, 19,  1 ],
	       [  5, 19,  1 ],
	       [  6, 19,  1 ],
	       [  7, 19,  1 ],
	       [  9, 19,  1 ],
	       [ 11, 19,  1 ],
	       [ 16, 19,  1 ],
	       [ 17, 19,  1 ],
	       [  2, 19, -1 ],
	       [  3, 19, -1 ],
	       [  8, 19, -1 ],
	       [ 10, 19, -1 ],
	       [ 12, 19, -1 ],
	       [ 13, 19, -1 ],
	       [ 14, 19, -1 ],
	       [ 15, 19, -1 ],
	       [ 18, 19, -1 ]) {
  foreach my $fun (\&jacobi, \&kronecker) {
    ok (&$fun ($$i[0], $$i[1]) == $$i[2]);

    ok (&$fun ($$i[0],      str($$i[1])) == $$i[2]);
    ok (&$fun (str($$i[0]),     $$i[1])  == $$i[2]);
    ok (&$fun (str($$i[0]), str($$i[1])) == $$i[2]);

    ok (&$fun ($$i[0],      mpz($$i[1])) == $$i[2]);
    ok (&$fun (mpz($$i[0]), $$i[1]) == $$i[2]);
    ok (&$fun (mpz($$i[0]), mpz($$i[1])) == $$i[2]);
  }
}

#------------------------------------------------------------------------------
# GMP::Mpz::lcm

ok (lcm (2) == 2);
ok (lcm (0) == 0);
ok (lcm (0,0) == 0);
ok (lcm (0,0,0) == 0);
ok (lcm (0,0,0,0) == 0);
ok (lcm (2,0) == 0);
ok (lcm (-2,0) == 0);
ok (lcm (2,3) == 6);
ok (lcm (2,3,4) == 12);
ok (lcm (2,-3) == 6);
ok (lcm (-2,3) == 6);
ok (lcm (-2,-3) == 6);
ok (lcm (mpz(2)**512,1) == mpz(2)**512);
ok (lcm (mpz(2)**512,-1) == mpz(2)**512);
ok (lcm (-mpz(2)**512,1) == mpz(2)**512);
ok (lcm (-mpz(2)**512,-1) == mpz(2)**512);
ok (lcm (mpz(2)**512,mpz(2)**512) == mpz(2)**512);
ok (lcm (mpz(2)**512,-mpz(2)**512) == mpz(2)**512);
ok (lcm (-mpz(2)**512,mpz(2)**512) == mpz(2)**512);
ok (lcm (-mpz(2)**512,-mpz(2)**512) == mpz(2)**512);

#------------------------------------------------------------------------------
# GMP::Mpz::lucnum

ok (lucnum(0) == 2);
ok (lucnum(1) == 1);
ok (lucnum(2) == 3);
ok (lucnum(3) == 4);
ok (lucnum(4) == 7);
ok (lucnum(5) == 11);
ok (lucnum(6) == 18);

#------------------------------------------------------------------------------
# GMP::Mpz::lucnum2

{ my ($a, $b) = lucnum2(0); ok($a==2);  ok($b==-1); }
{ my ($a, $b) = lucnum2(1); ok($a==1);  ok($b==2); }
{ my ($a, $b) = lucnum2(2); ok($a==3);  ok($b==1); }
{ my ($a, $b) = lucnum2(3); ok($a==4);  ok($b==3); }
{ my ($a, $b) = lucnum2(4); ok($a==7);  ok($b==4); }
{ my ($a, $b) = lucnum2(5); ok($a==11); ok($b==7); }
{ my ($a, $b) = lucnum2(6); ok($a==18); ok($b==11); }

#------------------------------------------------------------------------------
# GMP::Mpz::nextprime

ok (nextprime(2) == 3);
ok (nextprime(3) == 5);
ok (nextprime(5) == 7);
ok (nextprime(7) == 11);
ok (nextprime(11) == 13);

#------------------------------------------------------------------------------
# GMP::Mpz::perfect_power_p

# ok (  perfect_power_p(mpz(-27)));
# ok (! perfect_power_p(mpz(-9)));
# ok (! perfect_power_p(mpz(-1)));
ok (  perfect_power_p(mpz(0)));
ok (  perfect_power_p(mpz(1)));
ok (! perfect_power_p(mpz(2)));
ok (! perfect_power_p(mpz(3)));
ok (  perfect_power_p(mpz(4)));
ok (  perfect_power_p(mpz(9)));
ok (  perfect_power_p(mpz(27)));
ok (  perfect_power_p(mpz(81)));

#------------------------------------------------------------------------------
# GMP::Mpz::perfect_square_p

ok (! perfect_square_p(mpz(-9)));
ok (! perfect_square_p(mpz(-1)));
ok (  perfect_square_p(mpz(0)));
ok (  perfect_square_p(mpz(1)));
ok (! perfect_square_p(mpz(2)));
ok (! perfect_square_p(mpz(3)));
ok (  perfect_square_p(mpz(4)));
ok (  perfect_square_p(mpz(9)));
ok (! perfect_square_p(mpz(27)));
ok (  perfect_square_p(mpz(81)));

#------------------------------------------------------------------------------
# GMP::Mpz::popcount

ok (popcount(7) == 3);

#------------------------------------------------------------------------------
# GMP::Mpz::powm

ok (powm (3,2,8) == 1);

#------------------------------------------------------------------------------
# GMP::Mpz::probab_prime_p

ok (  probab_prime_p(89,1));
ok (! probab_prime_p(81,1));

#------------------------------------------------------------------------------
# GMP::Mpz::realloc

{ my $z = mpz(123);
  realloc ($z, 512); }

#------------------------------------------------------------------------------
# GMP::Mpz::remove

{
  my ($rem, $mult);
  ($rem, $mult) = remove(12,3);
  ok ($rem == 4);
  ok ($mult == 1);
  ($rem, $mult) = remove(12,2);
  ok ($rem == 3);
  ok ($mult == 2);
}

#------------------------------------------------------------------------------
# GMP::Mpz::root

ok (root(0,2) == 0);
ok (root(8,3) == 2);
ok (root(-8,3) == -2);
ok (root(81,4) == 3);
ok (root(243,5) == 3);

#------------------------------------------------------------------------------
# GMP::Mpz::roote

{ my ($r,$e);
  ($r, $e) = roote(0,2);
  ok ($r == 0);
  ok ($e);
  ($r, $e) = roote(81,4);
  ok ($r == 3);
  ok ($e);
  ($r, $e) = roote(85,4);
  ok ($r == 3);
  ok (! $e);
}

#------------------------------------------------------------------------------
# GMP::Mpz::rootrem

{ my ($root, $rem) = rootrem (mpz(0), 1);
  ok ($root == 0); ok ($rem == 0); }
{ my ($root, $rem) = rootrem (mpz(0), 2);
  ok ($root == 0); ok ($rem == 0); }
{ my ($root, $rem) = rootrem (mpz(64), 2);
  ok ($root == 8); ok ($rem == 0); }
{ my ($root, $rem) = rootrem (mpz(64), 3);
  ok ($root == 4); ok ($rem == 0); }
{ my ($root, $rem) = rootrem (mpz(65), 3);
  ok ($root == 4); ok ($rem == 1); }

#------------------------------------------------------------------------------
# GMP::Mpz::scan0

ok (scan0 (0, 0) == 0);
ok (scan0 (1, 0) == 1);
ok (scan0 (3, 0) == 2);
ok (scan0 (-1, 0) == ~0);
ok (scan0 (-2, 1) == ~0);

#------------------------------------------------------------------------------
# GMP::Mpz::scan1

ok (scan1 (1, 0) == 0);
ok (scan1 (2, 0) == 1);
ok (scan1 (4, 0) == 2);
ok (scan1 (0, 0) == ~0);
ok (scan1 (3, 2) == ~0);

#------------------------------------------------------------------------------
# GMP::Mpz::setbit

{ my $a = mpz(3); setbit ($a, 1); ok ($a == 3); }
{ my $a = mpz(3); setbit ($a, 2); ok ($a == 7); }

{ my $a = 3; setbit ($a, 1); ok ($a == 3); }
{ my $a = 3; setbit ($a, 2); ok ($a == 7); }

# mutate only given variable
{ my $a = mpz(0);
  my $b = $a;
  setbit ($a, 0);
  ok ($a == 1);
  ok ($b == 0);
}
{ my $a = 0;
  my $b = $a;
  setbit ($a, 0);
  ok ($a == 1);
  ok ($b == 0);
}

{ tie my $a, 'Mytie', mpz(3);
  setbit ($a, 2);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 7);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}
{ tie my $a, 'Mytie', 3;
  setbit ($a, 2);
  ok ($Mytie::fetched > 0);    # used fetch
  ok ($Mytie::stored > 0);     # used store
  ok ($a == 7);                # expected result
  ok (UNIVERSAL::isa($a,"GMP::Mpz"));
  ok (tied($a));               # still tied
}

{ my $b = mpz(2);
  tie my $a, 'Mytie', $b;
  setbit ($a, 0);
  ok ($a == 3);
  ok ($b == 2);
  ok (tied($a));
}
{ my $b = 2;
  tie my $a, 'Mytie', $b;
  setbit ($a, 0);
  ok ($a == 3);
  ok ($b == 2);
  ok (tied($a));
}

#------------------------------------------------------------------------------
# GMP::Mpz::sizeinbase

ok (sizeinbase(1,10) == 1);
ok (sizeinbase(100,10) == 3);
ok (sizeinbase(9999,10) == 5);

#------------------------------------------------------------------------------
# GMP::Mpz::sqrtrem

{
  my ($root, $rem) = sqrtrem(mpz(0));
  ok ($root == 0);
  ok ($rem == 0);
}
{
  my ($root, $rem) = sqrtrem(mpz(1));
  ok ($root == 1);
  ok ($rem == 0);
}
{
  my ($root, $rem) = sqrtrem(mpz(2));
  ok ($root == 1);
  ok ($rem == 1);
}
{
  my ($root, $rem) = sqrtrem(mpz(9));
  ok ($root == 3);
  ok ($rem == 0);
}
{
  my ($root, $rem) = sqrtrem(mpz(35));
  ok ($root == 5);
  ok ($rem == 10);
}
{
  my ($root, $rem) = sqrtrem(mpz(0));
  ok ($root == 0);
  ok ($rem == 0);
}

#------------------------------------------------------------------------------
# GMP::Mpz::tdiv

{ my ($q, $r);
  ($q, $r) = tdiv (16, 3);
  ok ($q == 5);
  ok ($r == 1);
  ($q, $r) = tdiv (16, -3);
  ok ($q == -5);
  ok ($r == 1);
  ($q, $r) = tdiv (-16, 3);
  ok ($q == -5);
  ok ($r == -1);
  ($q, $r) = tdiv (-16, -3);
  ok ($q == 5);
  ok ($r == -1);
}

#------------------------------------------------------------------------------
# GMP::Mpz::tdiv_2exp

{ my ($q, $r);
  ($q, $r) = tdiv_2exp (23, 2);
  ok ($q == 5);
  ok ($r == 3);
  ($q, $r) = tdiv_2exp (-23, 2);
  ok ($q == -5);
  ok ($r == -3);
}

#------------------------------------------------------------------------------
# GMP::Mpz::tstbit

ok (tstbit (6, 0) == 0);
ok (tstbit (6, 1) == 1);
ok (tstbit (6, 2) == 1);
ok (tstbit (6, 3) == 0);




#------------------------------------------------------------------------------
# GMP::Mpq

#------------------------------------------------------------------------------
# GMP::Mpq::new

ok (mpq(0) == 0);
ok (mpq('0') == 0);
ok (mpq(substr('101',1,1)) == 0);
ok (mpq(0.0) == 0);
ok (mpq(mpz(0)) == 0);
ok (mpq(mpq(0)) == 0);
ok (mpq(mpf(0)) == 0);

{ tie my $t, 'Mytie', 0; ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', '0'; ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', substr('101',1,1); ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', 0.0; ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', mpz(0); ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', mpq(0); ok (mpq($t) == 0); }
{ tie my $t, 'Mytie', mpf(0); ok (mpq($t) == 0); }

ok (mpq(-123) == -123);
ok (mpq('-123') == -123);
ok (mpq(substr('1-1231',1,4)) == -123);
ok (mpq(-123.0) == -123);
ok (mpq(mpz(-123)) == -123);
ok (mpq(mpq(-123)) == -123);
ok (mpq(mpf(-123)) == -123);

{ tie my $t, 'Mytie', -123; ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', '-123'; ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', substr('1-1231',1,4); ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', -123.0; ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', mpz(-123); ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', mpq(-123); ok (mpq($t) == -123); }
{ tie my $t, 'Mytie', mpf(-123); ok (mpq($t) == -123); }

ok (mpq($ivnv_2p128) == $str_2p128);
{ tie my $t, 'Mytie', $ivnv_2p128; ok (mpq($t) == $str_2p128); }

ok (mpq('3/2') == mpq(3,2));
ok (mpq('3/1') == mpq(3,1));
ok (mpq('-3/2') == mpq(-3,2));
ok (mpq('-3/1') == mpq(-3,1));
ok (mpq('0x3') == mpq(3,1));
ok (mpq('0b111') == mpq(7,1));
ok (mpq('0b0') == mpq(0,1));

ok (mpq($uv_max) > 0);
ok (mpq($uv_max) == mpq($uv_max_str));
{ tie my $t, 'Mytie', $uv_max; ok (mpq($t) > 0); }
{ tie my $t, 'Mytie', $uv_max; ok (mpq($t) == mpq($uv_max_str)); }

{ my $x = 123.5;
  kill (0, $x);
  ok (mpq($x) == 123.5);
  tie my $t, 'Mytie', $x;
  ok (mpq($t) == 123.5);
}

#------------------------------------------------------------------------------
# GMP::Mpq::overload_abs

ok (abs(mpq(0)) == 0);
ok (abs(mpq(123)) == 123);
ok (abs(mpq(-123)) == 123);

{ my $x = mpq(-123); $x = abs($x); ok ($x == 123); }
{ my $x = mpq(0);    $x = abs($x); ok ($x == 0);   }
{ my $x = mpq(123);  $x = abs($x); ok ($x == 123); }

{ tie my $t, 'Mytie', mpq(0); ok (abs($t) == 0); }
{ tie my $t, 'Mytie', mpq(123); ok (abs($t) == 123); }
{ tie my $t, 'Mytie', mpq(-123); ok (abs($t) == 123); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_add

ok (mpq(0) + 1 == 1);
ok (mpq(-1) + 1 == 0);
ok (1 + mpq(0) == 1);
ok (1 + mpq(-1) == 0);

ok (mpq(1,2)+mpq(1,3) == mpq(5,6));
ok (mpq(1,2)+mpq(-1,3) == mpq(1,6));
ok (mpq(-1,2)+mpq(1,3) == mpq(-1,6));
ok (mpq(-1,2)+mpq(-1,3) == mpq(-5,6));

#------------------------------------------------------------------------------
# GMP::Mpq::overload_addeq

{ my $a = mpq(7); $a += 1; ok ($a == 8); }
{ my $a = mpq(7); my $b = $a; $a += 1; ok ($a == 8); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_bool

if (mpq(0))   { ok (0); } else { ok (1); }
if (mpq(123)) { ok (1); } else { ok (0); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_dec

{ my $a = mpq(0); ok ($a-- == 0); ok ($a == -1); }
{ my $a = mpq(0); ok (--$a == -1); }

{ my $a = mpq(0); my $b = $a; $a--; ok ($a == -1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_div

ok (mpq(6) / 2 == 3);
ok (mpq(-6) / 2 == -3);
ok (mpq(6) / -2 == -3);
ok (mpq(-6) / -2 == 3);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_diveq

{ my $a = mpq(21); $a /= 3; ok ($a == 7); }
{ my $a = mpq(21); my $b = $a; $a /= 3; ok ($a == 7); ok ($b == 21); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_eq

{ my $a = mpq(0);
  my $b = $a;
  $a = mpq(1);
  ok ($a == 1);
  ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_inc

{ my $a = mpq(0); ok ($a++ == 0); ok ($a == 1); }
{ my $a = mpq(0); ok (++$a == 1); }

{ my $a = mpq(0); my $b = $a; $a++; ok ($a == 1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_lshift

{ my $a = mpq(7) << 1; ok ($a == 14); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_lshifteq

{ my $a = mpq(7); $a <<= 1; ok ($a == 14); }
{ my $a = mpq(7); my $b = $a; $a <<= 1; ok ($a == 14); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_mul

ok (mpq(2) * 3 == 6);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_muleq

{ my $a = mpq(7); $a *= 3;  ok ($a == 21); }
{ my $a = mpq(7); my $b = $a; $a *= 3;  ok ($a == 21); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_neg

ok (- mpq(0) == 0);
ok (- mpq(123) == -123);
ok (- mpq(-123) == 123);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_not

if (not mpq(0))   { ok (1); } else { ok (0); }
if (not mpq(123)) { ok (0); } else { ok (1); }

ok ((! mpq(0)) == 1);
ok ((! mpq(123)) == 0);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_pow

ok (mpq(0) ** 1 == 0);
ok (mpq(1) ** 1 == 1);
ok (mpq(2) ** 0 == 1);
ok (mpq(2) ** 1 == 2);
ok (mpq(2) ** 2 == 4);
ok (mpq(2) ** 3 == 8);
ok (mpq(2) ** 4 == 16);

ok (mpq(0) ** mpq(1) == 0);
ok (mpq(1) ** mpq(1) == 1);
ok (mpq(2) ** mpq(0) == 1);
ok (mpq(2) ** mpq(1) == 2);
ok (mpq(2) ** mpq(2) == 4);
ok (mpq(2) ** mpq(3) == 8);
ok (mpq(2) ** mpq(4) == 16);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_poweq

{ my $a = mpq(3); $a **= 4; ok ($a == 81); }
{ my $a = mpq(3); my $b = $a; $a **= 4; ok ($a == 81); ok ($b == 3); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_rshift

{ my $a = mpq(32) >> 1; ok ($a == 16); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_rshifteq

{ my $a = mpq(32); $a >>= 1; ok ($a == 16); }
{ my $a = mpq(32); my $b = $a; $a >>= 1; ok ($a == 16); ok ($b == 32); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_spaceship

ok (mpq(0) < 1);
ok (mpq(0) > -1);

ok (mpq(0) != 1);
ok (mpq(0) != -1);
ok (mpq(1) != 0);
ok (mpq(1) != -1);
ok (mpq(-1) != 0);
ok (mpq(-1) != 1);

ok (mpq(3,2) > 1);
ok (mpq(3,2) < 2);

ok (mpq(0) < 1.0);
ok (mpq(0) < '1');
ok (mpq(0) < substr('-1',1,1));
ok (mpq(0) < mpz(1));
ok (mpq(0) < mpq(1));
ok (mpq(0) < mpf(1));
ok (mpq(0) < $uv_max);

#------------------------------------------------------------------------------
# GMP::Mpq::overload_string

{ my $x = mpq(0);    ok("$x" eq "0"); }
{ my $x = mpq(123);  ok("$x" eq "123"); }
{ my $x = mpq(-123); ok("$x" eq "-123"); }

{ my $q = mpq(5,7);  ok("$q" eq "5/7"); }
{ my $q = mpq(-5,7); ok("$q" eq "-5/7"); }

#------------------------------------------------------------------------------
# GMP::Mpq::overload_sub

ok (mpq(0) - 1 == -1);
ok (mpq(1) - 1 == 0);
ok (1 - mpq(0) == 1);
ok (1 - mpq(1) == 0);

ok (mpq(1,2)-mpq(1,3) == mpq(1,6));
ok (mpq(1,2)-mpq(-1,3) == mpq(5,6));
ok (mpq(-1,2)-mpq(1,3) == mpq(-5,6));
ok (mpq(-1,2)-mpq(-1,3) == mpq(-1,6));

#------------------------------------------------------------------------------
# GMP::Mpq::overload_subeq

{ my $a = mpq(7); $a -= 1; ok ($a == 6); }
{ my $a = mpq(7); my $b = $a; $a -= 1; ok ($a == 6); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpq::canonicalize

{ my $q = mpq(21,15); canonicalize($q);
  ok (num($q) == 7);
  ok (den($q) == 5);
}

#------------------------------------------------------------------------------
# GMP::Mpq::den

{ my $q = mpq(5,9); ok (den($q) == 9); }

#------------------------------------------------------------------------------
# GMP::Mpq::num

{ my $q = mpq(5,9); ok (num($q) == 5); }




#------------------------------------------------------------------------------
# GMP::Mpf

#------------------------------------------------------------------------------
# GMP::Mpf::new

ok (mpf(0) == 0);
ok (mpf('0') == 0);
ok (mpf(substr('101',1,1)) == 0);
ok (mpf(0.0) == 0);
ok (mpf(mpz(0)) == 0);
ok (mpf(mpq(0)) == 0);
ok (mpf(mpf(0)) == 0);

{ tie my $t, 'Mytie', 0; ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', '0'; ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', substr('101',1,1); ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', 0.0; ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', mpz(0); ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', mpq(0); ok (mpf($t) == 0); }
{ tie my $t, 'Mytie', mpf(0); ok (mpf($t) == 0); }

ok (mpf(-123) == -123);
ok (mpf('-123') == -123);
ok (mpf(substr('1-1231',1,4)) == -123);
ok (mpf(-123.0) == -123);
ok (mpf(mpz(-123)) == -123);
ok (mpf(mpq(-123)) == -123);
ok (mpf(mpf(-123)) == -123);

{ tie my $t, 'Mytie', -123; ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', '-123'; ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', substr('1-1231',1,4); ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', -123.0; ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', mpz(-123); ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', mpq(-123); ok (mpf($t) == -123); }
{ tie my $t, 'Mytie', mpf(-123); ok (mpf($t) == -123); }

ok (mpf($ivnv_2p128) == $str_2p128);
{ tie my $t, 'Mytie', $ivnv_2p128; ok (mpf($t) == $str_2p128); }

ok (mpf(-1.5) == -1.5);
ok (mpf(-1.0) == -1.0);
ok (mpf(-0.5) == -0.5);
ok (mpf(0) == 0);
ok (mpf(0.5) == 0.5);
ok (mpf(1.0) == 1.0);
ok (mpf(1.5) == 1.5);

ok (mpf("-1.5") == -1.5);
ok (mpf("-1.0") == -1.0);
ok (mpf("-0.5") == -0.5);
ok (mpf("0") == 0);
ok (mpf("0.5") == 0.5);
ok (mpf("1.0") == 1.0);
ok (mpf("1.5") == 1.5);

ok (mpf($uv_max) > 0);
ok (mpf($uv_max) == mpf($uv_max_str));
{ tie my $t, 'Mytie', $uv_max; ok (mpf($t) > 0); }
{ tie my $t, 'Mytie', $uv_max; ok (mpf($t) == mpf($uv_max_str)); }

{ my $x = 123.5;
  kill (0, $x);
  ok (mpf($x) == 123.5);
  tie my $t, 'Mytie', $x;
  ok (mpf($t) == 123.5);
}

#------------------------------------------------------------------------------
# GMP::Mpf::overload_abs

ok (abs(mpf(0)) == 0);
ok (abs(mpf(123)) == 123);
ok (abs(mpf(-123)) == 123);

{ my $x = mpf(-123); $x = abs($x); ok ($x == 123); }
{ my $x = mpf(0);    $x = abs($x); ok ($x == 0);   }
{ my $x = mpf(123);  $x = abs($x); ok ($x == 123); }

{ tie my $t, 'Mytie', mpf(0); ok (abs($t) == 0); }
{ tie my $t, 'Mytie', mpf(123); ok (abs($t) == 123); }
{ tie my $t, 'Mytie', mpf(-123); ok (abs($t) == 123); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_add

ok (mpf(0) + 1 == 1);
ok (mpf(-1) + 1 == 0);
ok (1 + mpf(0) == 1);
ok (1 + mpf(-1) == 0);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_addeq

{ my $a = mpf(7); $a += 1; ok ($a == 8); }
{ my $a = mpf(7); my $b = $a; $a += 1; ok ($a == 8); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_bool

if (mpf(0))   { ok (0); } else { ok (1); }
if (mpf(123)) { ok (1); } else { ok (0); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_dec

{ my $a = mpf(0); ok ($a-- == 0); ok ($a == -1); }
{ my $a = mpf(0); ok (--$a == -1); }

{ my $a = mpf(0); my $b = $a; $a--; ok ($a == -1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_div

ok (mpf(6) / 2 == 3);
ok (mpf(-6) / 2 == -3);
ok (mpf(6) / -2 == -3);
ok (mpf(-6) / -2 == 3);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_diveq

{ my $a = mpf(21); $a /= 3; ok ($a == 7); }
{ my $a = mpf(21); my $b = $a; $a /= 3; ok ($a == 7); ok ($b == 21); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_eq

{ my $a = mpf(0);
  my $b = $a;
  $a = mpf(1);
  ok ($a == 1);
  ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_inc

{ my $a = mpf(0); ok ($a++ == 0); ok ($a == 1); }
{ my $a = mpf(0); ok (++$a == 1); }

{ my $a = mpf(0); my $b = $a; $a++; ok ($a == 1); ok ($b == 0); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_lshift

{ my $a = mpf(7) << 1; ok ($a == 14); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_lshifteq

{ my $a = mpf(7); $a <<= 1; ok ($a == 14); }
{ my $a = mpf(7); my $b = $a; $a <<= 1; ok ($a == 14); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_mul

ok (mpf(2) * 3 == 6);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_muleq

{ my $a = mpf(7); $a *= 3;  ok ($a == 21); }
{ my $a = mpf(7); my $b = $a; $a *= 3;  ok ($a == 21); ok ($b == 7); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_neg

ok (- mpf(0) == 0);
ok (- mpf(123) == -123);
ok (- mpf(-123) == 123);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_not

if (not mpf(0))   { ok (1); } else { ok (0); }
if (not mpf(123)) { ok (0); } else { ok (1); }

ok ((! mpf(0)) == 1);
ok ((! mpf(123)) == 0);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_pow

ok (mpf(0) ** 1 == 0);
ok (mpf(1) ** 1 == 1);
ok (mpf(2) ** 0 == 1);
ok (mpf(2) ** 1 == 2);
ok (mpf(2) ** 2 == 4);
ok (mpf(2) ** 3 == 8);
ok (mpf(2) ** 4 == 16);

ok (mpf(0) ** mpf(1) == 0);
ok (mpf(1) ** mpf(1) == 1);
ok (mpf(2) ** mpf(0) == 1);
ok (mpf(2) ** mpf(1) == 2);
ok (mpf(2) ** mpf(2) == 4);
ok (mpf(2) ** mpf(3) == 8);
ok (mpf(2) ** mpf(4) == 16);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_poweq

{ my $a = mpf(3); $a **= 4; ok ($a == 81); }
{ my $a = mpf(3); my $b = $a; $a **= 4; ok ($a == 81); ok ($b == 3); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_rshift

{ my $a = mpf(32) >> 1; ok ($a == 16); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_rshifteq

{ my $a = mpf(32); $a >>= 1; ok ($a == 16); }
{ my $a = mpf(32); my $b = $a; $a >>= 1; ok ($a == 16); ok ($b == 32); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_sqrt

ok (sqrt(mpf(0)) == 0);
ok (sqrt(mpf(1)) == 1);
ok (sqrt(mpf(4)) == 2);
ok (sqrt(mpf(81)) == 9);

ok (sqrt(mpf(0.25)) == 0.5);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_spaceship

ok (mpf(0) < 1);
ok (mpf(0) > -1);

ok (mpf(0) != 1);
ok (mpf(0) != -1);
ok (mpf(1) != 0);
ok (mpf(1) != -1);
ok (mpf(-1) != 0);
ok (mpf(-1) != 1);

ok (mpf(0) < 1.0);
ok (mpf(0) < '1');
ok (mpf(0) < substr('-1',1,1));
ok (mpf(0) < mpz(1));
ok (mpf(0) < mpq(1));
ok (mpf(0) < mpf(1));
ok (mpf(0) < $uv_max);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_string

{ my $x = mpf(0);    ok ("$x" eq "0"); }
{ my $x = mpf(123);  ok ("$x" eq "123"); }
{ my $x = mpf(-123); ok ("$x" eq "-123"); }

{ my $f = mpf(0.25);   	 ok ("$f" eq "0.25"); }
{ my $f = mpf(-0.25);  	 ok ("$f" eq "-0.25"); }
{ my $f = mpf(1.25);   	 ok ("$f" eq "1.25"); }
{ my $f = mpf(-1.25);  	 ok ("$f" eq "-1.25"); }
{ my $f = mpf(1000000);	 ok ("$f" eq "1000000"); }
{ my $f = mpf(-1000000); ok ("$f" eq "-1000000"); }

#------------------------------------------------------------------------------
# GMP::Mpf::overload_sub

ok (mpf(0) - 1 == -1);
ok (mpf(1) - 1 == 0);
ok (1 - mpf(0) == 1);
ok (1 - mpf(1) == 0);

#------------------------------------------------------------------------------
# GMP::Mpf::overload_subeq

{ my $a = mpf(7); $a -= 1; ok ($a == 6); }
{ my $a = mpf(7); my $b = $a; $a -= 1; ok ($a == 6); ok ($b == 7); }


#------------------------------------------------------------------------------
# GMP::Mpf::ceil

ok (ceil (mpf(-7.5)) == -7.0);
ok (ceil (mpf(7.5)) == 8.0);

#------------------------------------------------------------------------------
# GMP::Mpf::floor

ok (floor(mpf(-7.5)) == -8.0);
ok (floor(mpf(7.5)) == 7.0);

#------------------------------------------------------------------------------
# GMP::Mpf::mpf_eq

{ my $old_prec = get_default_prec();
  set_default_prec(128);

  ok (  mpf_eq (mpz("0x10000000000000001"), mpz("0x10000000000000002"), 1));
  ok (! mpf_eq (mpz("0x11"), mpz("0x12"), 128));

  set_default_prec($old_prec);
}

#------------------------------------------------------------------------------
# GMP::Mpf::get_default_prec

get_default_prec();

#------------------------------------------------------------------------------
# GMP::Mpf::get_prec

{ my $x = mpf(1.0, 512);
  ok (get_prec ($x) == 512);
}

#------------------------------------------------------------------------------
# GMP::Mpf::reldiff

ok (reldiff (2,4) == 1);
ok (reldiff (4,2) == 0.5);

#------------------------------------------------------------------------------
# GMP::Mpf::set_default_prec

{ my $old_prec = get_default_prec();

  set_default_prec(512);
  ok (get_default_prec () == 512);

  set_default_prec($old_prec);
}

#------------------------------------------------------------------------------
# GMP::Mpf::set_prec

{ my $x = mpf(1.0, 512);
  my $y = $x;
  set_prec ($x, 1024);
  ok (get_prec ($x) == 1024);
  ok (get_prec ($y) == 512);
}

#------------------------------------------------------------------------------
# GMP::Mpf::trunc

ok (trunc(mpf(-7.5)) == -7.0);
ok (trunc(mpf(7.5)) == 7.0);



#------------------------------------------------------------------------------
# GMP::Rand

#------------------------------------------------------------------------------
# GMP::Rand::new

{ my $r = randstate();                          ok (defined $r); }
{ my $r = randstate('lc_2exp', 1, 2, 3);        ok (defined $r); }
{ my $r = randstate('lc_2exp_size', 64);        ok (defined $r); }
{ my $r = randstate('lc_2exp_size', 999999999); ok (! defined $r); }
{ my $r = randstate('mt');                      ok (defined $r); }

{ # copying a randstate results in same sequence
  my $r1 = randstate('lc_2exp_size', 64);
  $r1->seed(123);
  my $r2 = randstate($r1);
  for (1 .. 20) {
    my $z1 = mpz_urandomb($r1, 20);
    my $z2 = mpz_urandomb($r2, 20);
    ok ($z1 == $z2);
  }
}

#------------------------------------------------------------------------------
# GMP::Rand::seed

{ my $r = randstate();
  $r->seed(123);
  $r->seed(time());
}

#------------------------------------------------------------------------------
# GMP::Rand::mpf_urandomb

{ my $r = randstate();
  my $f = mpf_urandomb($r,1024);
  ok (UNIVERSAL::isa($f,"GMP::Mpf")); }

#------------------------------------------------------------------------------
# GMP::Rand::mpz_urandomb

{ my $r = randstate();
  my $z = mpz_urandomb($r, 1024);
  ok (UNIVERSAL::isa($z,"GMP::Mpz")); }

#------------------------------------------------------------------------------
# GMP::Rand::mpz_rrandomb

{ my $r = randstate();
  my $z = mpz_rrandomb($r, 1024);
  ok (UNIVERSAL::isa($z,"GMP::Mpz")); }

#------------------------------------------------------------------------------
# GMP::Rand::mpz_urandomm

{ my $r = randstate();
  my $z = mpz_urandomm($r, mpz(3)**100);
  ok (UNIVERSAL::isa($z,"GMP::Mpz")); }

#------------------------------------------------------------------------------
# GMP::Rand::mpz_urandomb_ui

{ my $r = randstate();
  foreach (1 .. 20) {
    my $u = gmp_urandomb_ui($r,8);
    ok ($u >= 0);
    ok ($u < 256);
  }
}

#------------------------------------------------------------------------------
# GMP::Rand::mpz_urandomm_ui

{ my $r = randstate();
  foreach (1 .. 20) {
    my $u = gmp_urandomm_ui($r,8);
    ok ($u >= 0);
    ok ($u < 8);
  }
}




#------------------------------------------------------------------------------
# GMP module

#------------------------------------------------------------------------------
# GMP::fits_slong_p

ok (GMP::fits_slong_p(0));

# in perl 5.005 uv_max is only 32-bits on a 64-bit system, so won't exceed a
# long
# ok (! GMP::fits_slong_p($uv_max));

ok (GMP::fits_slong_p(0.0));

ok (GMP::fits_slong_p('0'));

ok (GMP::fits_slong_p(substr('999999999999999999999999999999',1,1)));

ok (! mpz("-9999999999999999999999999999999999999999999")->fits_slong_p());
ok (  mpz(-123)->fits_slong_p());
ok (  mpz(0)->fits_slong_p());
ok (  mpz(123)->fits_slong_p());
ok (! mpz("9999999999999999999999999999999999999999999")->fits_slong_p());

ok (! mpq("-9999999999999999999999999999999999999999999")->fits_slong_p());
ok (  mpq(-123)->fits_slong_p());
ok (  mpq(0)->fits_slong_p());
ok (  mpq(123)->fits_slong_p());
ok (! mpq("9999999999999999999999999999999999999999999")->fits_slong_p());

ok (! mpf("-9999999999999999999999999999999999999999999")->fits_slong_p());
ok (  mpf(-123)->fits_slong_p());
ok (  mpf(0)->fits_slong_p());
ok (  mpf(123)->fits_slong_p());
ok (! mpf("9999999999999999999999999999999999999999999")->fits_slong_p());

#------------------------------------------------------------------------------
# GMP::get_d

ok (GMP::get_d(123) == 123.0);

ok (GMP::get_d($uv_max) > 0);

ok (GMP::get_d(123.0) == 123.0);

ok (GMP::get_d('123') == 123.0);

ok (GMP::get_d(mpz(123)) == 123.0);

ok (GMP::get_d(mpq(123)) == 123.0);

ok (GMP::get_d(mpf(123)) == 123.0);

#------------------------------------------------------------------------------
# GMP::get_d_2exp

{ my ($dbl, $exp) = get_d_2exp (0);
  ok ($dbl == 0); ok ($exp == 0); }
{ my ($dbl, $exp) = get_d_2exp (1);
  ok ($dbl == 0.5); ok ($exp == 1); }

{ my ($dbl, $exp) = get_d_2exp ($uv_max);
  ok ($dbl > 0.0); ok ($exp > 0); }

{ my ($dbl, $exp) = get_d_2exp (0.5);
  ok ($dbl == 0.5); ok ($exp == 0); }
{ my ($dbl, $exp) = get_d_2exp (0.25);
  ok ($dbl == 0.5); ok ($exp == -1); }

{ my ($dbl, $exp) = get_d_2exp ("1.0");
  ok ($dbl == 0.5); ok ($exp == 1); }

{ my ($dbl, $exp) = get_d_2exp (mpz ("256"));
  ok ($dbl == 0.5); ok ($exp == 9); }

{ my ($dbl, $exp) = get_d_2exp (mpq ("1/16"));
  ok ($dbl == 0.5); ok ($exp == -3); }

{ my ($dbl, $exp) = get_d_2exp (mpf ("1.5"));
  ok ($dbl == 0.75); ok ($exp == 1); }
{ my ($dbl, $exp) = get_d_2exp (mpf ("3.0"));
  ok ($dbl == 0.75); ok ($exp == 2); }

#------------------------------------------------------------------------------
# GMP::get_str

ok (get_str(-123) eq '-123');
ok (get_str('-123') eq '-123');
ok (get_str(substr('x-123x',1,4)) eq '-123');
ok (get_str(mpz(-123)) eq '-123');
ok (get_str(mpq(-123)) eq '-123');

ok (get_str(-123,10) eq '-123');
ok (get_str('-123',10) eq '-123');
ok (get_str(substr('x-123x',1,4),10) eq '-123');
ok (get_str(mpz(-123),10) eq '-123');
ok (get_str(mpq(-123),10) eq '-123');

ok (get_str(-123,16) eq '-7b');
ok (get_str('-123',16) eq '-7b');
ok (get_str(substr('x-123x',1,4),16) eq '-7b');
ok (get_str(mpz(-123),16) eq '-7b');
ok (get_str(mpq(-123),16) eq '-7b');

ok (get_str(-123,-16) eq '-7B');
ok (get_str('-123',-16) eq '-7B');
ok (get_str(substr('x-123x',1,4),-16) eq '-7B');
ok (get_str(mpz(-123),-16) eq '-7B');
ok (get_str(mpq(-123),-16) eq '-7B');

# is a float in past versions of perl without UV type
{ my ($str, $exp) = get_str($uv_max);
  ok ($str eq $uv_max_str); }

ok (get_str(mpq(5/8)) eq "5/8");
ok (get_str(mpq(-5/8)) eq "-5/8");
ok (get_str(mpq(255/256),16) eq "ff/100");
ok (get_str(mpq(255/256),-16) eq "FF/100");
ok (get_str(mpq(-255/256),16) eq "-ff/100");
ok (get_str(mpq(-255/256),-16) eq "-FF/100");

{ my ($s,$e) = get_str(1.5, 10);      ok ($s eq '15'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(1.5), 10); ok ($s eq '15'); ok ($e == 1); }

{ my ($s,$e) = get_str(-1.5, 10);      ok ($s eq '-15'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(-1.5), 10); ok ($s eq '-15'); ok ($e == 1); }

{ my ($s,$e) = get_str(1.5, 16);      ok ($s eq '18'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(1.5), 16); ok ($s eq '18'); ok ($e == 1); }

{ my ($s,$e) = get_str(-1.5, 16);      ok ($s eq '-18'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(-1.5), 16); ok ($s eq '-18'); ok ($e == 1); }

{ my ($s,$e) = get_str(65536.0, 16);      ok ($s eq '1'); ok ($e == 5); }
{ my ($s,$e) = get_str(mpf(65536.0), 16); ok ($s eq '1'); ok ($e == 5); }

{ my ($s,$e) = get_str(1.625, 16);      ok ($s eq '1a'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(1.625), 16); ok ($s eq '1a'); ok ($e == 1); }

{ my ($s,$e) = get_str(1.625, -16);      ok ($s eq '1A'); ok ($e == 1); }
{ my ($s,$e) = get_str(mpf(1.625), -16); ok ($s eq '1A'); ok ($e == 1); }

{ my ($s, $e) = get_str(255.0,16,0);      ok ($s eq "ff"); ok ($e == 2); }
{ my ($s, $e) = get_str(mpf(255.0),16,0); ok ($s eq "ff"); ok ($e == 2); }

{ my ($s, $e) = get_str(255.0,-16,0);      ok ($s eq "FF"); ok ($e == 2); }
{ my ($s, $e) = get_str(mpf(255.0),-16,0); ok ($s eq "FF"); ok ($e == 2); }

#------------------------------------------------------------------------------
# GMP::get_si

ok (GMP::get_si(123) == 123.0);

# better not assume anything about the relatives sizes of long and UV
ok (GMP::get_si($uv_max) != 0);

ok (GMP::get_si(123.0) == 123.0);

ok (GMP::get_si('123') == 123.0);

ok (GMP::get_si(mpz(123)) == 123.0);

ok (GMP::get_si(mpq(123)) == 123.0);

ok (GMP::get_si(mpf(123)) == 123.0);

#------------------------------------------------------------------------------
# GMP::integer_p

ok (  GMP::integer_p (0));
ok (  GMP::integer_p (123));
ok (  GMP::integer_p (-123));

ok (  GMP::integer_p ($uv_max));

ok (  GMP::integer_p (0.0));
ok (  GMP::integer_p (123.0));
ok (  GMP::integer_p (-123.0));
ok (! GMP::integer_p (0.5));
ok (! GMP::integer_p (123.5));
ok (! GMP::integer_p (-123.5));

ok (  GMP::integer_p ('0'));
ok (  GMP::integer_p ('123'));
ok (  GMP::integer_p ('-123'));
ok (! GMP::integer_p ('0.5'));
ok (! GMP::integer_p ('123.5'));
ok (! GMP::integer_p ('-123.5'));
ok (! GMP::integer_p ('5/8'));

ok (  GMP::integer_p (mpz(1)));

ok (  GMP::integer_p (mpq(1)));
ok (! GMP::integer_p (mpq(1,2)));

ok (  GMP::integer_p (mpf(1.0)));
ok (! GMP::integer_p (mpf(1.5)));

#------------------------------------------------------------------------------
# GMP::odd_p

ok (! odd_p(0));
ok (  odd_p(1));
ok (! odd_p(2));

ok (  odd_p($uv_max));

ok (  odd_p(mpz(-3)));
ok (! odd_p(mpz(-2)));
ok (  odd_p(mpz(-1)));
ok (! odd_p(mpz(0)));
ok (  odd_p(mpz(1)));
ok (! odd_p(mpz(2)));
ok (  odd_p(mpz(3)));

#------------------------------------------------------------------------------
# GMP::printf

GMP::printf ("hello world\n");

sub via_printf {
  my $s;
  open TEMP, ">test.tmp" or die;
  GMP::printf TEMP @_;
  close TEMP or die;
  open TEMP, "<test.tmp" or die;
  read (TEMP, $s, 1024);
  close TEMP or die;
  unlink 'test.tmp';
  return $s;
}

ok (sprintf ("%d", mpz(123)) eq '123');
ok (sprintf ("%d %d %d", 456, mpz(123), 789) eq '456 123 789');
ok (sprintf ("%d", mpq(15,16)) eq '15/16');
ok (sprintf ("%f", mpf(1.5)) eq '1.500000');
ok (sprintf ("%.2f", mpf(1.5)) eq '1.50');

ok (sprintf ("%*d", 6, 123) eq '   123');
ok (sprintf ("%*d", 6, mpz(123))  eq '   123');
ok (sprintf ("%*d", 6, mpq(15,16))  eq ' 15/16');

ok (sprintf ("%x", 123) eq '7b');
ok (sprintf ("%x", mpz(123))  eq '7b');
ok (sprintf ("%X", 123) eq '7B');
ok (sprintf ("%X", mpz(123))  eq '7B');
ok (sprintf ("%#x", 123) eq '0x7b');
ok (sprintf ("%#x", mpz(123))  eq '0x7b');
ok (sprintf ("%#X", 123) eq '0X7B');
ok (sprintf ("%#X", mpz(123))  eq '0X7B');

ok (sprintf ("%x", mpq(15,16))  eq 'f/10');
ok (sprintf ("%X", mpq(15,16))  eq 'F/10');
ok (sprintf ("%#x", mpq(15,16))  eq '0xf/0x10');
ok (sprintf ("%#X", mpq(15,16))  eq '0XF/0X10');

ok (sprintf ("%*.*f", 10, 3, 1.25) eq '     1.250');
ok (sprintf ("%*.*f", 10, 3, mpf(1.5))   eq '     1.500');

ok (via_printf ("%d", mpz(123)) eq '123');
ok (via_printf ("%d %d %d", 456, mpz(123), 789) eq '456 123 789');
ok (via_printf ("%d", mpq(15,16)) eq '15/16');
ok (via_printf ("%f", mpf(1.5)) eq '1.500000');
ok (via_printf ("%.2f", mpf(1.5)) eq '1.50');

ok (via_printf ("%*d", 6, 123) eq '   123');
ok (via_printf ("%*d", 6, mpz(123))  eq '   123');
ok (via_printf ("%*d", 6, mpq(15,16))  eq ' 15/16');

ok (via_printf ("%x", 123) eq '7b');
ok (via_printf ("%x", mpz(123))  eq '7b');
ok (via_printf ("%X", 123) eq '7B');
ok (via_printf ("%X", mpz(123))  eq '7B');
ok (via_printf ("%#x", 123) eq '0x7b');
ok (via_printf ("%#x", mpz(123))  eq '0x7b');
ok (via_printf ("%#X", 123) eq '0X7B');
ok (via_printf ("%#X", mpz(123))  eq '0X7B');

ok (via_printf ("%x", mpq(15,16))  eq 'f/10');
ok (via_printf ("%X", mpq(15,16))  eq 'F/10');
ok (via_printf ("%#x", mpq(15,16))  eq '0xf/0x10');
ok (via_printf ("%#X", mpq(15,16))  eq '0XF/0X10');

ok (via_printf ("%*.*f", 10, 3, 1.25) eq '     1.250');
ok (via_printf ("%*.*f", 10, 3, mpf(1.5))   eq '     1.500');

#------------------------------------------------------------------------------
# GMP::sgn

ok (sgn(-123) == -1);
ok (sgn(0)    == 0);
ok (sgn(123)  == 1);

ok (sgn($uv_max) == 1);

ok (sgn(-123.0) == -1);
ok (sgn(0.0)    == 0);
ok (sgn(123.0)  == 1);

ok (sgn('-123') == -1);
ok (sgn('0')    == 0);
ok (sgn('123')  == 1);
ok (sgn('-123.0') == -1);
ok (sgn('0.0')    == 0);
ok (sgn('123.0')  == 1);

ok (sgn(substr('x-123x',1,4)) == -1);
ok (sgn(substr('x0x',1,1))    == 0);
ok (sgn(substr('x123x',1,3))  == 1);

ok (mpz(-123)->sgn() == -1);
ok (mpz(0)   ->sgn() == 0);
ok (mpz(123) ->sgn() == 1);

ok (mpq(-123)->sgn() == -1);
ok (mpq(0)   ->sgn() == 0);
ok (mpq(123) ->sgn() == 1);

ok (mpf(-123)->sgn() == -1);
ok (mpf(0)   ->sgn() == 0);
ok (mpf(123) ->sgn() == 1);



#------------------------------------------------------------------------------
# overloaded constants

if ($] > 5.00503) {
  if (! do 'test2.pl') {
    die "Cannot run test2.pl\n";
  }
}




#------------------------------------------------------------------------------
# $# stuff
#
# For some reason "local $#" doesn't leave $# back at its default undefined
# state when exiting the block.

{ local $# = 'hi %.0f there';
  my $f = mpf(123);
  ok ("$f" eq 'hi 123 there'); }



# Local variables:
# perl-indent-level: 2
# End:
