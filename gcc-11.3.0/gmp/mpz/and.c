/* mpz_and -- Logical and.

Copyright 1991, 1993, 1994, 1996, 1997, 2000, 2001, 2003, 2005, 2012 Free
Software Foundation, Inc.

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

#include "gmp.h"
#include "gmp-impl.h"

void
mpz_and (mpz_ptr res, mpz_srcptr op1, mpz_srcptr op2)
{
  mp_srcptr op1_ptr, op2_ptr;
  mp_size_t op1_size, op2_size;
  mp_ptr res_ptr;
  mp_size_t res_size;
  mp_size_t i;
  TMP_DECL;

  TMP_MARK;
  op1_size = SIZ(op1);
  op2_size = SIZ(op2);

  op1_ptr = PTR(op1);
  op2_ptr = PTR(op2);

  if (op1_size >= 0)
    {
      if (op2_size >= 0)
	{
	  res_size = MIN (op1_size, op2_size);
	  /* First loop finds the size of the result.  */
	  for (i = res_size - 1; i >= 0; i--)
	    if ((op1_ptr[i] & op2_ptr[i]) != 0)
	      break;
	  res_size = i + 1;

	  /* Handle allocation, now then we know exactly how much space is
	     needed for the result.  */
	  res_ptr = MPZ_REALLOC (res, res_size);
	  /* Don't re-read op1_ptr and op2_ptr.  Since res_size <=
	     MIN(op1_size, op2_size), res is not changed when op1
	     is identical to res or op2 is identical to res.  */

	  SIZ(res) = res_size;
	  if (LIKELY (res_size != 0))
	    mpn_and_n (res_ptr, op1_ptr, op2_ptr, res_size);
	  return;
	}
      else /* op2_size < 0 */
	{
	  /* Fall through to the code at the end of the function.  */
	}
    }
  else
    {
      if (op2_size < 0)
	{
	  mp_ptr opx, opy;
	  mp_limb_t cy;

	  /* Both operands are negative, so will be the result.
	     -((-OP1) & (-OP2)) = -(~(OP1 - 1) & ~(OP2 - 1)) =
	     = ~(~(OP1 - 1) & ~(OP2 - 1)) + 1 =
	     = ((OP1 - 1) | (OP2 - 1)) + 1      */

	  /* It might seem as we could end up with an (invalid) result with
	     a leading zero-limb here when one of the operands is of the
	     type 1,,0,,..,,.0.  But some analysis shows that we surely
	     would get carry into the zero-limb in this situation...  */

	  op1_size = -op1_size;
	  op2_size = -op2_size;

	  if (op1_size > op2_size)
	    MPN_SRCPTR_SWAP (op1_ptr, op1_size, op2_ptr, op2_size);

	  TMP_ALLOC_LIMBS_2 (opx, op1_size, opy, op2_size);
	  mpn_sub_1 (opx, op1_ptr, op1_size, (mp_limb_t) 1);
	  op1_ptr = opx;

	  mpn_sub_1 (opy, op2_ptr, op2_size, (mp_limb_t) 1);
	  op2_ptr = opy;

	  res_ptr = MPZ_REALLOC (res, 1 + op2_size);
	  /* Don't re-read OP1_PTR and OP2_PTR.  They point to temporary
	     space--never to the space PTR(res) used to point to before
	     reallocation.  */

	  MPN_COPY (res_ptr + op1_size, op2_ptr + op1_size,
		    op2_size - op1_size);
	  mpn_ior_n (res_ptr, op1_ptr, op2_ptr, op1_size);
	  res_size = op2_size;

	  cy = mpn_add_1 (res_ptr, res_ptr, res_size, (mp_limb_t) 1);
	  res_ptr[res_size] = cy;
	  res_size += (cy != 0);

	  SIZ(res) = -res_size;
	  TMP_FREE;
	  return;
	}
      else
	{
	  /* We should compute -OP1 & OP2.  Swap OP1 and OP2 and fall
	     through to the code that handles OP1 & -OP2.  */
	  MPN_SRCPTR_SWAP (op1_ptr, op1_size, op2_ptr, op2_size);
	}

    }

  {
#if ANDNEW
    mp_size_t op2_lim;
    mp_size_t count;

    /* OP2 must be negated as with infinite precision.

       Scan from the low end for a non-zero limb.  The first non-zero
       limb is simply negated (two's complement).  Any subsequent
       limbs are one's complemented.  Of course, we don't need to
       handle more limbs than there are limbs in the other, positive
       operand as the result for those limbs is going to become zero
       anyway.  */

    /* Scan for the least significant non-zero OP2 limb, and zero the
       result meanwhile for those limb positions.  (We will surely
       find a non-zero limb, so we can write the loop with one
       termination condition only.)  */
    for (i = 0; op2_ptr[i] == 0; i++)
      res_ptr[i] = 0;
    op2_lim = i;

    op2_size = -op2_size;

    if (op1_size <= op2_size)
      {
	/* The ones-extended OP2 is >= than the zero-extended OP1.
	   RES_SIZE <= OP1_SIZE.  Find the exact size.  */
	for (i = op1_size - 1; i > op2_lim; i--)
	  if ((op1_ptr[i] & ~op2_ptr[i]) != 0)
	    break;
	res_size = i + 1;
	for (i = res_size - 1; i > op2_lim; i--)
	  res_ptr[i] = op1_ptr[i] & ~op2_ptr[i];
	res_ptr[op2_lim] = op1_ptr[op2_lim] & -op2_ptr[op2_lim];
	/* Yes, this *can* happen!  */
	MPN_NORMALIZE (res_ptr, res_size);
      }
    else
      {
	/* The ones-extended OP2 is < than the zero-extended OP1.
	   RES_SIZE == OP1_SIZE, since OP1 is normalized.  */
	res_size = op1_size;
	MPN_COPY (res_ptr + op2_size, op1_ptr + op2_size, op1_size - op2_size);
	for (i = op2_size - 1; i > op2_lim; i--)
	  res_ptr[i] = op1_ptr[i] & ~op2_ptr[i];
	res_ptr[op2_lim] = op1_ptr[op2_lim] & -op2_ptr[op2_lim];
      }

    SIZ(res) = res_size;
#else

    /* OP1 is positive and zero-extended,
       OP2 is negative and ones-extended.
       The result will be positive.
       OP1 & -OP2 = OP1 & ~(OP2 - 1).  */

    mp_ptr opx;

    op2_size = -op2_size;
    opx = TMP_ALLOC_LIMBS (op2_size);
    mpn_sub_1 (opx, op2_ptr, op2_size, (mp_limb_t) 1);
    op2_ptr = opx;

    if (op1_size > op2_size)
      {
	/* The result has the same size as OP1, since OP1 is normalized
	   and longer than the ones-extended OP2.  */
	res_size = op1_size;

	/* Handle allocation, now then we know exactly how much space is
	   needed for the result.  */
	res_ptr = MPZ_REALLOC (res, res_size);
	/* Don't re-read OP1_PTR or OP2_PTR.  Since res_size = op1_size,
	   op1 is not changed if it is identical to res.
	   OP2_PTR points to temporary space.  */

	MPN_COPY (res_ptr + op2_size, op1_ptr + op2_size, res_size - op2_size);
	mpn_andn_n (res_ptr, op1_ptr, op2_ptr, op2_size);

	SIZ(res) = res_size;
      }
    else
      {
	/* Find out the exact result size.  Ignore the high limbs of OP2,
	   OP1 is zero-extended and would make the result zero.  */
	for (i = op1_size - 1; i >= 0; i--)
	  if ((op1_ptr[i] & ~op2_ptr[i]) != 0)
	    break;
	res_size = i + 1;

	/* Handle allocation, now then we know exactly how much space is
	   needed for the result.  */
	res_ptr = MPZ_REALLOC (res, res_size);
	/* Don't re-read OP1_PTR.  Since res_size <= op1_size,
	   op1 is not changed if it is identical to res.
	   Don't re-read OP2_PTR.  It points to temporary space--never
	   to the space PTR(res) used to point to before reallocation.  */

	if (LIKELY (res_size != 0))
	  mpn_andn_n (res_ptr, op1_ptr, op2_ptr, res_size);

	SIZ(res) = res_size;
      }
#endif
  }
  TMP_FREE;
}
