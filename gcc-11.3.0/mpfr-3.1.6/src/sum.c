/* Sum -- efficiently sum a list of floating-point numbers

Copyright 2004-2017 Free Software Foundation, Inc.
Contributed by the AriC and Caramba projects, INRIA.

This file is part of the GNU MPFR Library.

The GNU MPFR Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The GNU MPFR Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MPFR Library; see the file COPYING.LESSER.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA. */

/* Reference: James Demmel and Yozo Hida, Fast and accurate floating-point
   summation with application to computational geometry, Numerical Algorithms,
   volume 37, number 1-4, pages 101--112, 2004. */

/* Note about the 3.1 branch and earlier: the "const" in the prototypes of
   mpfr_sum and related functions is in an incorrect position. This bug is
   present in the source only (since r3642); the MPFR manual is correct.
   This is fixed in the trunk for the future MPFR 4.0.0. Let's not change
   the 3.1 branch as it may be seen as an ABI breakage and this bug does
   not have any consequence for the API.
*/

#define MPFR_NEED_LONGLONG_H
#include "mpfr-impl.h"

/* I would really like to use "mpfr_srcptr const []" but the norm is buggy:
   it doesn't automaticaly cast a "mpfr_ptr []" to "mpfr_srcptr const []"
   if necessary. So the choice are:
     mpfr_s **                : ok
     mpfr_s *const*           : ok
     mpfr_s **const           : ok
     mpfr_s *const*const      : ok
     const mpfr_s *const*     : no
     const mpfr_s **const     : no
     const mpfr_s *const*const: no
   VL: this is not a bug, but a feature. See the reason here:
     http://c-faq.com/ansi/constmismatch.html
*/
static void heap_sort (mpfr_srcptr *const, unsigned long, mpfr_srcptr *);
static void count_sort (mpfr_srcptr *const, unsigned long, mpfr_srcptr *,
                        mpfr_exp_t, mpfr_uexp_t);

/* Either sort the tab in perm and returns 0
   Or returns 1 for +INF, -1 for -INF and 2 for NAN.
   Also set *maxprec to the maximal precision of tab[0..n-1] and of the
   initial value of *maxprec.
*/
int
mpfr_sum_sort (mpfr_srcptr *const tab, unsigned long n, mpfr_srcptr *perm,
               mpfr_prec_t *maxprec)
{
  mpfr_exp_t min, max;
  mpfr_uexp_t exp_num;
  unsigned long i;
  int sign_inf;

  sign_inf = 0;
  min = MPFR_EMIN_MAX;
  max = MPFR_EMAX_MIN;
  for (i = 0; i < n; i++)
    {
      if (MPFR_UNLIKELY (MPFR_IS_SINGULAR (tab[i])))
        {
          if (MPFR_IS_NAN (tab[i]))
            return 2; /* Return NAN code */
          else if (MPFR_IS_INF (tab[i]))
            {
              if (sign_inf == 0) /* No previous INF */
                sign_inf = MPFR_SIGN (tab[i]);
              else if (sign_inf != MPFR_SIGN (tab[i]))
                return 2; /* Return NAN */
            }
        }
      else
        {
          MPFR_ASSERTD (MPFR_IS_PURE_FP (tab[i]));
          if (MPFR_GET_EXP (tab[i]) < min)
            min = MPFR_GET_EXP(tab[i]);
          if (MPFR_GET_EXP (tab[i]) > max)
            max = MPFR_GET_EXP(tab[i]);
        }
      if (MPFR_PREC (tab[i]) > *maxprec)
        *maxprec = MPFR_PREC (tab[i]);
    }
  if (MPFR_UNLIKELY (sign_inf != 0))
    return sign_inf;

  exp_num = max - min + 1;
  /* FIXME : better test */
  if (exp_num > n * MPFR_INT_CEIL_LOG2 (n))
    heap_sort (tab, n, perm);
  else
    count_sort (tab, n, perm, min, exp_num);
  return 0;
}

#define GET_EXP1(x) (MPFR_IS_ZERO (x) ? min : MPFR_GET_EXP (x))
/* Performs a count sort of the entries */
static void
count_sort (mpfr_srcptr *const tab, unsigned long n,
            mpfr_srcptr *perm, mpfr_exp_t min, mpfr_uexp_t exp_num)
{
  unsigned long *account;
  unsigned long target_rank, i;
  MPFR_TMP_DECL(marker);

  /* Reserve a place for potential 0 (with EXP min-1)
     If there is no zero, we only lose one unused entry */
  min--;
  exp_num++;

  /* Performs a counting sort of the entries */
  MPFR_TMP_MARK (marker);
  account = (unsigned long *) MPFR_TMP_ALLOC (exp_num * sizeof *account);
  for (i = 0; i < exp_num; i++)
    account[i] = 0;
  for (i = 0; i < n; i++)
    account[GET_EXP1 (tab[i]) - min]++;
  for (i = exp_num - 1; i >= 1; i--)
    account[i - 1] += account[i];
  for (i = 0; i < n; i++)
    {
      target_rank = --account[GET_EXP1 (tab[i]) - min];
      perm[target_rank] = tab[i];
    }
  MPFR_TMP_FREE (marker);
}


#define GET_EXP2(x) (MPFR_IS_ZERO (x) ? MPFR_EMIN_MIN : MPFR_GET_EXP (x))

/* Performs a heap sort of the entries */
static void
heap_sort (mpfr_srcptr *const tab, unsigned long n, mpfr_srcptr *perm)
{
  unsigned long dernier_traite;
  unsigned long i, pere;
  mpfr_srcptr tmp;
  unsigned long fils_gauche, fils_droit, fils_indigne;
  /* Reminder of a heap structure :
     node(i) has for left son node(2i +1) and right son node(2i)
     and father(node(i)) = node((i - 1) / 2)
  */

  /* initialize the permutation to identity */
  for (i = 0; i < n; i++)
    perm[i] = tab[i];

  /* insertion phase */
  for (dernier_traite = 1; dernier_traite < n; dernier_traite++)
    {
      i = dernier_traite;
      while (i > 0)
        {
          pere = (i - 1) / 2;
          if (GET_EXP2 (perm[pere]) > GET_EXP2 (perm[i]))
            {
              tmp = perm[pere];
              perm[pere] = perm[i];
              perm[i] = tmp;
              i = pere;
            }
          else
            break;
        }
    }

  /* extraction phase */
  for (dernier_traite = n - 1; dernier_traite > 0; dernier_traite--)
    {
      tmp = perm[0];
      perm[0] = perm[dernier_traite];
      perm[dernier_traite] = tmp;

      i = 0;
      while (1)
        {
          fils_gauche = 2 * i + 1;
          fils_droit = fils_gauche + 1;
          if (fils_gauche < dernier_traite)
            {
              if (fils_droit < dernier_traite)
                {
                  if (GET_EXP2(perm[fils_droit]) < GET_EXP2(perm[fils_gauche]))
                    fils_indigne = fils_droit;
                  else
                    fils_indigne = fils_gauche;

                  if (GET_EXP2 (perm[i]) > GET_EXP2 (perm[fils_indigne]))
                    {
                      tmp = perm[i];
                      perm[i] = perm[fils_indigne];
                      perm[fils_indigne] = tmp;
                      i = fils_indigne;
                    }
                  else
                    break;
                }
              else /* on a un fils gauche, pas de fils droit */
                {
                  if (GET_EXP2 (perm[i]) > GET_EXP2 (perm[fils_gauche]))
                    {
                      tmp = perm[i];
                      perm[i] = perm[fils_gauche];
                      perm[fils_gauche] = tmp;
                    }
                  break;
                }
            }
          else /* on n'a pas de fils */
            break;
        }
    }
}


/* Sum a list of float with order given by permutation perm,
 * intermediate size set to F. Return non-zero if at least one of
 * the operations is inexact (thus 0 implies that the sum is exact).
 * Internal use function.
 */
static int
sum_once (mpfr_ptr ret, mpfr_srcptr *const tab, unsigned long n, mpfr_prec_t F)
{
  mpfr_t sum;
  unsigned long i;
  int error_trap;

  MPFR_ASSERTD (n >= 2);

  mpfr_init2 (sum, F);
  error_trap = mpfr_set (sum, tab[0], MPFR_RNDN);
  for (i = 1; i < n - 1; i++)
    {
      MPFR_ASSERTD (!MPFR_IS_NAN (sum) && !MPFR_IS_INF (sum));
      if (mpfr_add (sum, sum, tab[i], MPFR_RNDN))
        error_trap = 1;
    }
  if (mpfr_add (ret, sum, tab[n - 1], MPFR_RNDN))
    error_trap = 1;
  mpfr_clear (sum);
  return error_trap;
}

/* Sum a list of floating-point numbers.
 * If the return value is 0, then the sum is exact.
 * Otherwise the return value gives no information.
 */
int
mpfr_sum (mpfr_ptr ret, mpfr_ptr *const tab_p, unsigned long n, mpfr_rnd_t rnd)
{
  mpfr_t cur_sum;
  mpfr_prec_t prec;
  mpfr_srcptr *perm, *const tab = (mpfr_srcptr *) tab_p;
  int k, error_trap;
  MPFR_ZIV_DECL (loop);
  MPFR_SAVE_EXPO_DECL (expo);
  MPFR_TMP_DECL (marker);

  if (MPFR_UNLIKELY (n <= 1))
    {
      if (n < 1)
        {
          MPFR_SET_ZERO (ret);
          MPFR_SET_POS (ret);
          return 0;
        }
      else
        return mpfr_set (ret, tab[0], rnd);
    }

  /* Sort and treat special cases */
  MPFR_TMP_MARK (marker);
  perm = (mpfr_srcptr *) MPFR_TMP_ALLOC (n * sizeof *perm);
  prec = MPFR_PREC (ret);
  error_trap = mpfr_sum_sort (tab, n, perm, &prec);
  /* Check if there was a NAN or a INF */
  if (MPFR_UNLIKELY (error_trap != 0))
    {
      MPFR_TMP_FREE (marker);
      if (error_trap == 2)
        {
          MPFR_SET_NAN (ret);
          MPFR_RET_NAN;
        }
      MPFR_SET_INF (ret);
      MPFR_SET_SIGN (ret, error_trap);
      MPFR_RET (0);
    }

  /* Initial precision is max(prec(ret),prec(tab[0]),...,prec(tab[n-1])) */
  k = MPFR_INT_CEIL_LOG2 (n) + 1;
  prec += k + 2;
  mpfr_init2 (cur_sum, prec);

  /* Ziv Loop */
  MPFR_SAVE_EXPO_MARK (expo);
  MPFR_ZIV_INIT (loop, prec);
  for (;;)
    {
      error_trap = sum_once (cur_sum, perm, n, prec + k);
      if (MPFR_LIKELY (error_trap == 0 ||
                       (!MPFR_IS_ZERO (cur_sum) &&
                        mpfr_can_round (cur_sum, prec - 2,
                                        MPFR_RNDN, rnd, MPFR_PREC (ret)))))
        break;
      MPFR_ZIV_NEXT (loop, prec);
      mpfr_set_prec (cur_sum, prec);
    }
  MPFR_ZIV_FREE (loop);
  MPFR_TMP_FREE (marker);

  if (mpfr_set (ret, cur_sum, rnd))
    error_trap = 1;
  mpfr_clear (cur_sum);

  MPFR_SAVE_EXPO_FREE (expo);
  if (mpfr_check_range (ret, 0, rnd))
    error_trap = 1;
  return error_trap; /* It doesn't return the ternary value */
}

/* __END__ */
