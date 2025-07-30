/* Test file for mpfr_log.

Copyright 1999, 2001-2017 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>

#include "mpfr-test.h"

#ifdef CHECK_EXTERNAL
static int
test_log (mpfr_ptr a, mpfr_srcptr b, mpfr_rnd_t rnd_mode)
{
  int res;
  int ok = rnd_mode == MPFR_RNDN && mpfr_number_p (b) && mpfr_get_prec (a)>=53;
  if (ok)
    {
      mpfr_print_raw (b);
    }
  res = mpfr_log (a, b, rnd_mode);
  if (ok)
    {
      printf (" ");
      mpfr_print_raw (a);
      printf ("\n");
    }
  return res;
}
#else
#define test_log mpfr_log
#endif

static void
check2 (const char *as, mpfr_rnd_t rnd_mode, const char *res1s)
{
  mpfr_t ta, tres;

  mpfr_inits2 (53, ta, tres, (mpfr_ptr) 0);
  mpfr_set_str1 (ta, as);
  test_log (tres, ta, rnd_mode);

  if (mpfr_cmp_str1 (tres, res1s))
    {
      printf ("mpfr_log failed for    a=%s, rnd_mode=%s\n",
              as, mpfr_print_rnd_mode (rnd_mode));
      printf ("correct result is        %s\n mpfr_log gives          ",
              res1s);
      mpfr_out_str(stdout, 10, 0, tres, MPFR_RNDN);
      exit (1);
    }
  mpfr_clears (ta, tres, (mpfr_ptr) 0);
}

static void
check3 (double d, unsigned long prec, mpfr_rnd_t rnd)
{
  mpfr_t x, y;

  mpfr_init2 (x, prec);
  mpfr_init2 (y, prec);
  mpfr_set_d (x, d, rnd);
  test_log (y, x, rnd);
  mpfr_out_str (stdout, 10, 0, y, rnd);
  puts ("");
  mpfr_print_binary (y);
  puts ("");
  mpfr_clear (x);
  mpfr_clear (y);
}

/* examples from Jean-Michel Muller and Vincent Lefevre
   Cf http://www.ens-lyon.fr/~jmmuller/Intro-to-TMD.htm
*/

static void
check_worst_cases (void)
{
  check2("1.00089971802309629645", MPFR_RNDD, "8.99313519443722736088e-04");
  check2("1.00089971802309629645", MPFR_RNDN, "8.99313519443722844508e-04");
  check2("1.00089971802309629645", MPFR_RNDU, "8.99313519443722844508e-04");

  check2("1.01979300812244555452", MPFR_RNDD, "1.95996734891603630047e-02");
  check2("1.01979300812244555452", MPFR_RNDN, "1.95996734891603664741e-02");
  check2("1.01979300812244555452", MPFR_RNDU, "1.95996734891603664741e-02");

  check2("1.02900871924604464525", MPFR_RNDD, "2.85959303301472726744e-02");
  check2("1.02900871924604464525", MPFR_RNDN, "2.85959303301472761438e-02");
  check2("1.02900871924604464525", MPFR_RNDU, "2.85959303301472761438e-02");

  check2("1.27832870030418943585", MPFR_RNDD, "2.45553521871417795852e-01");
  check2("1.27832870030418943585", MPFR_RNDN, "2.45553521871417823608e-01");
  check2("1.27832870030418943585", MPFR_RNDU, "2.45553521871417823608e-01");

  check2("1.31706530746788241792", MPFR_RNDD, "2.75406009586277422674e-01");
  check2("1.31706530746788241792", MPFR_RNDN, "2.75406009586277478185e-01");
  check2("1.31706530746788241792", MPFR_RNDU, "2.75406009586277478185e-01");

  check2("1.47116981099449883885", MPFR_RNDD, "3.86057874110010412760e-01");
  check2("1.47116981099449883885", MPFR_RNDN, "3.86057874110010412760e-01");
  check2("1.47116981099449883885", MPFR_RNDU, "3.86057874110010468272e-01");

  check2("1.58405446812987782401", MPFR_RNDD, "4.59987679246663727639e-01");
  check2("1.58405446812987782401", MPFR_RNDN, "4.59987679246663783150e-01");
  check2("1.58405446812987782401", MPFR_RNDU, "4.59987679246663783150e-01");

  check2("1.67192331263391547047", MPFR_RNDD, "5.13974647961076613889e-01");
  check2("1.67192331263391547047", MPFR_RNDN, "5.13974647961076724911e-01");
  check2("1.67192331263391547047", MPFR_RNDU, "5.13974647961076724911e-01");

  check2("1.71101198068990645318", MPFR_RNDD, "5.37084997042120315669e-01");
  check2("1.71101198068990645318", MPFR_RNDN, "5.37084997042120315669e-01");
  check2("1.71101198068990645318", MPFR_RNDU, "5.37084997042120426691e-01");

  check2("1.72634853551388700588", MPFR_RNDD, "5.46008504786553605648e-01");
  check2("1.72634853551388700588", MPFR_RNDN, "5.46008504786553716670e-01");
  check2("1.72634853551388700588", MPFR_RNDU, "5.46008504786553716670e-01");

  check2("2.00028876593004323325", MPFR_RNDD, "6.93291553102749702475e-01");
  check2("2.00028876593004323325", MPFR_RNDN, "6.93291553102749813497e-01");
  check2("2.00028876593004323325", MPFR_RNDU, "6.93291553102749813497e-01");

  check2("6.27593230200363105808", MPFR_RNDD, "1.83672204800630312072");
  check2("6.27593230200363105808", MPFR_RNDN, "1.83672204800630334276");
  check2("6.27593230200363105808", MPFR_RNDU, "1.83672204800630334276");

  check2("7.47216682321367997588", MPFR_RNDD, "2.01118502712453661729");
  check2("7.47216682321367997588", MPFR_RNDN, "2.01118502712453706138");
  check2("7.47216682321367997588", MPFR_RNDU, "2.01118502712453706138");

  check2("9.34589857718275318632", MPFR_RNDD, "2.23493759221664944903");
  check2("9.34589857718275318632", MPFR_RNDN, "2.23493759221664989312");
  check2("9.34589857718275318632", MPFR_RNDU, "2.23493759221664989312");

  check2("10.6856587560831854944", MPFR_RNDD, "2.36890253928838445674");
  check2("10.6856587560831854944", MPFR_RNDN, "2.36890253928838445674");
  check2("10.6856587560831854944", MPFR_RNDU, "2.36890253928838490083");

  check2("12.4646345033981766903", MPFR_RNDD, "2.52289539471636015122");
  check2("12.4646345033981766903", MPFR_RNDN, "2.52289539471636015122");
  check2("12.4646345033981766903", MPFR_RNDU, "2.52289539471636059531");

  check2("17.0953275851761752335", MPFR_RNDD, "2.83880518553861849185");
  check2("17.0953275851761752335", MPFR_RNDN, "2.83880518553861893594");
  check2("17.0953275851761752335", MPFR_RNDU, "2.83880518553861893594");

  check2("19.8509496207496916043", MPFR_RNDD, "2.98825184582516722998");
  check2("19.8509496207496916043", MPFR_RNDN, "2.98825184582516722998");
  check2("19.8509496207496916043", MPFR_RNDU, "2.98825184582516767406");

  check2("23.9512076062771335216", MPFR_RNDD, "3.17601874455977206679");
  check2("23.9512076062771335216", MPFR_RNDN, "3.17601874455977206679");
  check2("23.9512076062771335216", MPFR_RNDU, "3.17601874455977251088");

  check2("428.315247165198229595", MPFR_RNDD, "6.05985948325268264369");
  check2("428.315247165198229595", MPFR_RNDN, "6.05985948325268353187");
  check2("428.315247165198229595", MPFR_RNDU, "6.05985948325268353187");
}

static void
special (void)
{
  mpfr_t x, y;
  int inex;
  mpfr_exp_t emin, emax;

  emin = mpfr_get_emin ();
  emax = mpfr_get_emax ();

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);

  /* Check special case: An overflow in const_pi could occurs! */
  set_emin (-125);
  set_emax (128);
  mpfr_set_prec (y, 24*2);
  mpfr_set_prec (x, 24);
  mpfr_set_str_binary (x, "0.111110101010101011110101E0");
  test_log (y, x, MPFR_RNDN);
  set_emin (emin);
  set_emax (emax);

  mpfr_set_prec (y, 53);
  mpfr_set_prec (x, 53);
  mpfr_set_ui (x, 3, MPFR_RNDD);
  test_log (y, x, MPFR_RNDD);
  if (mpfr_cmp_str1 (y, "1.09861228866810956"))
    {
      printf ("Error in mpfr_log(3) for MPFR_RNDD\n");
      exit (1);
    }

  /* check large precision */
  mpfr_set_prec (x, 3322);
  mpfr_set_prec (y, 3322);
  mpfr_set_ui (x, 3, MPFR_RNDN);
  mpfr_sqrt (x, x, MPFR_RNDN);
  test_log (y, x, MPFR_RNDN);

  /* negative argument */
  mpfr_set_si (x, -1, MPFR_RNDN);
  test_log (y, x, MPFR_RNDN);
  MPFR_ASSERTN(mpfr_nan_p (y));

  /* infinite loop when  */
  set_emax (128);
  mpfr_set_prec (x, 251);
  mpfr_set_prec (y, 251);
  mpfr_set_str_binary (x, "0.10010111000000000001101E8");
  /* x = 4947981/32768, log(x) ~ 5.017282... */
  test_log (y, x, MPFR_RNDN);

  set_emax (emax);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  inex = test_log (y, x, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) < 0);

  mpfr_set_ui (x, 0, MPFR_RNDN);
  mpfr_neg (x, x, MPFR_RNDN);
  inex = test_log (y, x, MPFR_RNDN);
  MPFR_ASSERTN (inex == 0);
  MPFR_ASSERTN (mpfr_inf_p (y));
  MPFR_ASSERTN (mpfr_sgn (y) < 0);

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
x_near_one (void)
{
  mpfr_t x, y;
  int inex;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 16);

  mpfr_set_ui (x, 1, MPFR_RNDN);
  mpfr_nextbelow (x);
  inex = mpfr_log (y, x, MPFR_RNDD);
  if (mpfr_cmp_str (y, "-0.1000000000000001E-31", 2, MPFR_RNDN)
      || inex >= 0)
    {
      printf ("Failure in x_near_one, got inex = %d and\ny = ", inex);
      mpfr_dump (y);
    }

  mpfr_clears (x, y, (mpfr_ptr) 0);
}

#define TEST_FUNCTION test_log
#define TEST_RANDOM_POS 8
#include "tgeneric.c"

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  if (argc==4)
    {   /* tlog x prec rnd */
      check3 (atof(argv[1]), atoi(argv[2]), (mpfr_rnd_t) atoi(argv[3]));
      goto done;
    }

  special ();
  check_worst_cases();

  check2("1.01979300812244555452", MPFR_RNDN, "1.95996734891603664741e-02");
  check2("10.0",MPFR_RNDU,"2.30258509299404590110e+00");
  check2("6.0",MPFR_RNDU,"1.79175946922805517936");
  check2("1.0",MPFR_RNDZ,"0.0");
  check2("62.0",MPFR_RNDU,"4.12713438504509166905");
  check2("0.5",MPFR_RNDZ,"-6.93147180559945286226e-01");
  check2("3.0",MPFR_RNDZ,"1.09861228866810956006e+00");
  check2("234375765.0",MPFR_RNDU,"1.92724362186836231104e+01");
  check2("8.0",MPFR_RNDZ,"2.07944154167983574765e+00");
  check2("44.0",MPFR_RNDU,"3.78418963391826146392e+00");
  check2("1.01979300812244555452", MPFR_RNDN, "1.95996734891603664741e-02");

  /* bugs found by Vincent Lefe`vre */
  check2("0.99999599881598921769", MPFR_RNDN, "-0.0000040011920155404072924737977900999652547398000024259090423583984375");
  check2("9.99995576063808955247e-01",MPFR_RNDZ,"-4.42394597667932383816e-06");
  check2("9.99993687357856209097e-01",MPFR_RNDN,"-6.31266206860017342601e-06");
  check2("9.99995223520736886691e-01",MPFR_RNDN,"-4.77649067052670982220e-06");
  check2("9.99993025794720935551e-01",MPFR_RNDN,"-6.97422959894716163837e-06");
  check2("9.99987549017837484833e-01",MPFR_RNDN,"-1.24510596766369924330e-05");
  check2("9.99985901426543311032e-01",MPFR_RNDN,"-1.40986728425098585229e-05");
  check2("9.99986053947420794330e-01",MPFR_RNDN, "-0.000013946149826301084938555592540598837558718514628708362579345703125");
  check2("9.99971938247442126979e-01",MPFR_RNDN,"-2.80621462962173414790e-05");

  /* other bugs found by Vincent Lefe`vre */
  check2("1.18615436389927785905e+77",MPFR_RNDN,"1.77469768607706015473e+02");
  check2("9.48868723578399476187e+77",MPFR_RNDZ,"1.79549152432275803903e+02");
  check2("2.31822210096938820854e+89",MPFR_RNDN,"2.05770873832573869322e+02");

  /* further bugs found by Vincent Lefe`vre */
  check2("9.99999989485669482647e-01",MPFR_RNDZ,"-1.05143305726283042331e-08");
  check2("9.99999989237970177136e-01",MPFR_RNDZ,"-1.07620298807745377934e-08");
  check2("9.99999989239339082125e-01",MPFR_RNDN,"-1.07606609757704445430e-08");

  check2("7.3890560989306504",MPFR_RNDU,"2.0000000000000004"); /* exp(2.0) */
  check2("7.3890560989306495",MPFR_RNDU,"2.0"); /* exp(2.0) */
  check2("7.53428236571286402512e+34",MPFR_RNDZ,"8.03073567492226345621e+01");
  check2("6.18784121531737948160e+19",MPFR_RNDZ,"4.55717030391710693493e+01");
  check2("1.02560267603047283735e+00",MPFR_RNDD,"2.52804164149448735987e-02");
  check2("7.53428236571286402512e+34",MPFR_RNDZ,"8.03073567492226345621e+01");
  check2("1.42470900831881198052e+49",MPFR_RNDZ,"113.180637144887668910087086260318756103515625");

  check2("1.08013816255293777466e+11",MPFR_RNDN,"2.54055249841782604392e+01");
  check2("6.72783635300509015581e-37",MPFR_RNDU,"-8.32893948416799503320e+01");
  check2("2.25904918906057891180e-52",MPFR_RNDU,"-1.18919480823735682406e+02");
  check2("1.48901209246462951085e+00",MPFR_RNDD,"3.98112874867437460668e-01");
  check2("1.70322470467612341327e-01",MPFR_RNDN,"-1.77006175364294615626");
  check2("1.94572026316065240791e+01",MPFR_RNDD,"2.96821731676437838842");
  check2("4.01419512207026418764e+04",MPFR_RNDD,"1.06001772315501128218e+01");
  check2("9.47077365236487591672e-04",MPFR_RNDZ,"-6.96212977303956748187e+00");
  check2("3.95906157687589643802e-109",MPFR_RNDD,"-2.49605768114704119399e+02");
  check2("2.73874914516503004113e-02",MPFR_RNDD,"-3.59766888618655977794e+00");
  check2("9.18989072589566467669e-17",MPFR_RNDZ,"-3.69258425351464083519e+01");
  check2("7706036453608191045959753324430048151991964994788917248.0",MPFR_RNDZ,"126.3815989984199177342816255986690521240234375");
  check2("1.74827399630587801934e-23",MPFR_RNDZ,"-5.24008281254547156891e+01");
  check2("4.35302958401482307665e+22",MPFR_RNDD,"5.21277441046519527390e+01");
  check2("9.70791868689332915209e+00",MPFR_RNDD,"2.27294191194272210410e+00");
  check2("2.22183639799464011100e-01",MPFR_RNDN,"-1.50425103275253957413e+00");
  check2("2.27313466156682375540e+00",MPFR_RNDD,"8.21159787095675608448e-01");
  check2("6.58057413965851156767e-01",MPFR_RNDZ,"-4.18463096196088235600e-01");
  check2 ("7.34302197248998461006e+43",MPFR_RNDZ,"101.0049094695131799426235374994575977325439453125");
  check2("6.09969788341579732815e+00",MPFR_RNDD,"1.80823924264386204363e+00");

  x_near_one ();

  test_generic (2, 100, 40);

  data_check ("data/log", mpfr_log, "mpfr_log");
  bad_cases (mpfr_log, mpfr_exp, "mpfr_log", 256, -30, 30, 4, 128, 800, 50);

 done:
  tests_end_mpfr ();
  return 0;
}
