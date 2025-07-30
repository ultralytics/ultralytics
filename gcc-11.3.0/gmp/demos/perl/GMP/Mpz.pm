# GMP mpz module.

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


package GMP::Mpz;

require GMP;
require Exporter;
@ISA = qw(GMP Exporter);
@EXPORT = qw();
@EXPORT_OK = qw();
%EXPORT_TAGS = ('all' => [qw(
			     bin cdiv cdiv_2exp clrbit combit congruent_p
			     congruent_2exp_p divexact divisible_p
			     divisible_2exp_p even_p fac fdiv fdiv_2exp fib
			     fib2 gcd gcdext hamdist invert jacobi kronecker
			     lcm lucnum lucnum2 mod mpz mpz_export
			     mpz_import nextprime odd_p perfect_power_p
			     perfect_square_p popcount powm probab_prime_p
			     realloc remove root roote rootrem scan0 scan1
			     setbit sizeinbase sqrtrem tdiv tdiv_2exp
			     tstbit)],
		'constants'   => [@EXPORT],
		'noconstants' => [@EXPORT]);
Exporter::export_ok_tags('all');

use overload
    '+'    => \&overload_add,     '+='   => \&overload_addeq,
    '-'    => \&overload_sub,     '-='   => \&overload_subeq,
    '*'    => \&overload_mul,     '*='   => \&overload_muleq,
    '/'    => \&overload_div,     '/='   => \&overload_diveq,
    '%'    => \&overload_rem,     '%='   => \&overload_remeq,
    '<<'   => \&overload_lshift,  '<<='  => \&overload_lshifteq,
    '>>'   => \&overload_rshift,  '>>='  => \&overload_rshifteq,
    '**'   => \&overload_pow,     '**='  => \&overload_poweq,
    '&'    => \&overload_and,     '&='   => \&overload_andeq,
    '|'    => \&overload_ior,     '|='   => \&overload_ioreq,
    '^'    => \&overload_xor,     '^='   => \&overload_xoreq,

    'bool' => \&overload_bool,
    'not'  => \&overload_not,
    '!'    => \&overload_not,
    '~'    => \&overload_com,
    '<=>'  => \&overload_spaceship,
    '++'   => \&overload_inc,
    '--'   => \&overload_dec,
    '='    => \&overload_copy,
    'abs'  => \&overload_abs,
    'neg'  => \&overload_neg,
    'sqrt' => \&overload_sqrt,
    '""'   => \&overload_string;

sub import {
  foreach (@_) {
    if ($_ eq ':constants') {
      overload::constant ('integer' => \&overload_constant,
			  'binary'  => \&overload_constant,
			  'float'   => \&overload_constant);
    } elsif ($_ eq ':noconstants') {
      overload::remove_constant ('integer' => \&overload_constant,
				 'binary'  => \&overload_constant,
				 'float'   => \&overload_constant);
    }
  }
  goto &Exporter::import;
}

1;
__END__


# Local variables:
# perl-indent-level: 2
# End:
