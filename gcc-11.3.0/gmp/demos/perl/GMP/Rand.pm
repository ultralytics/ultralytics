# GMP random numbers module.

# Copyright 2001, 2003 Free Software Foundation, Inc.
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


package GMP::Rand;

require GMP;
require Exporter;
@ISA = qw(GMP Exporter);
@EXPORT = qw();
%EXPORT_TAGS = ('all' => [qw(
			     randstate mpf_urandomb mpz_rrandomb
			     mpz_urandomb mpz_urandomm gmp_urandomb_ui
			     gmp_urandomm_ui)]);
Exporter::export_ok_tags('all');
1;
__END__
