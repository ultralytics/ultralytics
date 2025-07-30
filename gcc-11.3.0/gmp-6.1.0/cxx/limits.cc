/* instantiation of numeric_limits specializations.

Copyright 2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */

#include "gmpxx.h"

namespace std {
#define GMPXX_INSTANTIATE_LIMITS(T) \
  const bool numeric_limits<T>::is_specialized; \
  const int  numeric_limits<T>::digits; \
  const int  numeric_limits<T>::digits10; \
  const int  numeric_limits<T>::max_digits10; \
  const bool numeric_limits<T>::is_signed; \
  const bool numeric_limits<T>::is_integer; \
  const bool numeric_limits<T>::is_exact; \
  const int  numeric_limits<T>::radix; \
  const int  numeric_limits<T>::min_exponent; \
  const int  numeric_limits<T>::min_exponent10; \
  const int  numeric_limits<T>::max_exponent; \
  const int  numeric_limits<T>::max_exponent10; \
  const bool numeric_limits<T>::has_infinity; \
  const bool numeric_limits<T>::has_quiet_NaN; \
  const bool numeric_limits<T>::has_signaling_NaN; \
  const float_denorm_style numeric_limits<T>::has_denorm; \
  const bool numeric_limits<T>::has_denorm_loss; \
  const bool numeric_limits<T>::is_iec559; \
  const bool numeric_limits<T>::is_bounded; \
  const bool numeric_limits<T>::is_modulo; \
  const bool numeric_limits<T>::traps; \
  const bool numeric_limits<T>::tinyness_before; \
  const float_round_style numeric_limits<T>::round_style

  GMPXX_INSTANTIATE_LIMITS(mpz_class);
  GMPXX_INSTANTIATE_LIMITS(mpq_class);
  GMPXX_INSTANTIATE_LIMITS(mpf_class);
}
