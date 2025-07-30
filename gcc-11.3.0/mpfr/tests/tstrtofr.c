/* Test file for mpfr_set_str.

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

#include <stdlib.h>

#include "mpfr-test.h"

static void
check_special (void)
{
  mpfr_t x, y;
  int res;
  char *s;

  mpfr_init (x);
  mpfr_init (y);

  /* Check dummy case */
  res = mpfr_strtofr (x, "1234567.89E1", NULL, 10, MPFR_RNDN);
  mpfr_set_str (y, "1234567.89E1", 10, MPFR_RNDN);
  if (mpfr_cmp (x, y))
    {
      printf ("Results differ between strtofr and set_str.\n"
              " set_str gives: ");
      mpfr_dump (y);
      printf (" strtofr gives: ");
      mpfr_dump (x);
      exit (1);
    }

  /* Check NAN  */
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NaN", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || *s != 0)
    {
      printf ("Error for setting NAN (1)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "+NaN", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || *s != 0)
    {
      printf ("Error for setting +NAN (1)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, " -NaN", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || *s != 0)
    {
      printf ("Error for setting -NAN (1)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "@nAn@xx", &s, 16, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "xx") )
    {
      printf ("Error for setting NAN (2)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NAN(abcdEDF__1256)Hello", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "Hello") )
    {
      printf ("Error for setting NAN (3)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NAN(abcdEDF)__1256)Hello", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "__1256)Hello") )
    {
      printf ("Error for setting NAN (4)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NAN(abc%dEDF)__1256)Hello", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "(abc%dEDF)__1256)Hello") )
    {
      printf ("Error for setting NAN (5)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NAN((abc))", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "((abc))") )
    {
      printf ("Error for setting NAN (6)\n");
      exit (1);
    }
  mpfr_set_ui (x, 0, MPFR_RNDN); /* make sure that x is modified */
  res = mpfr_strtofr (x, "NAN()foo", &s, 10, MPFR_RNDN);
  if (res != 0 || !mpfr_nan_p (x) || strcmp(s, "foo") )
    {
      printf ("Error for setting NAN (7)\n");
      exit (1);
    }

  /* Check INF */
  res = mpfr_strtofr (x, "INFINITY", &s, 8, MPFR_RNDN);
  if (res != 0 || !mpfr_inf_p (x) || *s != 0)
    {
      printf ("Error for setting INFINITY (1)\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  res = mpfr_strtofr (x, "INFANITY", &s, 8, MPFR_RNDN);
  if (res != 0 || !mpfr_inf_p (x) || strcmp(s, "ANITY"))
    {
      printf ("Error for setting INFINITY (2)\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  res = mpfr_strtofr (x, "@INF@*2", &s, 11, MPFR_RNDN);
  if (res != 0 || !mpfr_inf_p (x) || strcmp(s, "*2"))
    {
      printf ("Error for setting INFINITY (3)\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }

  /* Check Zero */
  res = mpfr_strtofr (x, " 00000", &s, 11, MPFR_RNDN);
  if (res != 0 || !mpfr_zero_p (x) || s[0] != 0)
    {
      printf ("Error for setting ZERO (1)\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }

  /* Check base 62 */
  res = mpfr_strtofr (x, "A", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 10))
    {
      printf ("Error for setting 'A' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "a", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 36))
    {
      printf ("Error for setting 'a' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "Z", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 35))
    {
      printf ("Error for setting 'Z' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "z", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 61))
    {
      printf ("Error for setting 'z' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "ZA", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 2180))
    {
      printf ("Error for setting 'ZA' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "za", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 3818))
    {
      printf ("Error for setting 'za' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "aZ", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 2267))
    {
      printf ("Error for setting 'aZ' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "Az", NULL, 62, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 681))
    {
      printf ("Error for setting 'Az' in base 62\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }

  /* Check base 60 */
  res = mpfr_strtofr (x, "Aa", NULL, 60, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 636))
    {
      printf ("Error for setting 'Aa' in base 60\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }
  res = mpfr_strtofr (x, "Zz", &s, 60, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 35) || strcmp(s, "z") )
    {
      printf ("Error for setting 'Zz' in base 60\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }

  /* Check base 61 */
  res = mpfr_strtofr (x, "z", &s, 61, MPFR_RNDN);
  if (res != 0 || mpfr_cmp_ui (x, 0) || strcmp(s, "z") )
    {
      printf ("Error for setting 'z' in base 61\n x=");
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN);
      putchar ('\n');
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

/* The following RefTable has been generated by this following code */
#if 0
#define MAX_NUM 100

int randomab (int a, int b)
{
  return a + rand () % (b-a);
}

int
main (void)
{
  int i, base;
  mpfr_t x;
  mpfr_prec_t p;
  mpfr_exp_t e;

  mpfr_init (x);
  printf ("struct dymmy_test { \n"
          " mpfr_prec_t prec; \n"
          " int base; \n"
          " const char *str; \n"
          " const char *binstr; \n"
          " } RefTable[] = { \n");
  for (i = 0 ; i < MAX_NUM ; i++)
    {
      p = randomab(2, 180);
      base = randomab (2, 30);
      e = randomab (-1<<15, 1<<15);
      mpfr_set_prec (x, p);
      mpfr_urandomb (x, RANDS);
      mpfr_mul_2si (x, x, e, MPFR_RNDN);
      printf("{%lu, %d,\n\"", p, base);
      mpfr_out_str (stdout, base, p, x, MPFR_RNDN);
      printf ("\",\n\"");
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN);
      printf ("\"}%c\n", i == MAX_NUM-1 ? ' ' : ',' );
    }
  printf("};\n");
  mpfr_clear (x);
}
#endif

static struct dymmy_test {
 mpfr_prec_t prec;
 int base;
 const char *str;
 const char *binstr;
 } RefTable[] = {
{39, 20,
"1.1c9jeh9jg12d8iiggf26b8ce2cig24agai51d9@1445",
"1.00111010111010001110110001101011101011e6245"},
{119, 3,
"1.2210112120221020220021000020101121120011021202212101222000011110211211122222001001221110102220122021121021101010120101e-5655",
"1.1111101110011110001101011100011000011100001011011100010011010010001000000111001010000001110111010100011000110010000000e-8963"},
{166, 18,
"3.ecg67g31434b74d8hhbe2dbbb46g9546cae72cae0cfghfh00ed7gebe9ca63b47h08bgbdeb880a76dea12he31e1ccd67e9dh22a911b46h517b745169b2g43egg2e4eah820cdb2132d6a4f9c63505dd4a0dafbc@-5946",
"1.011110010000110011111011111100110110010110000010100001101111111000010000011111110101100000010110011001100000010001100101000001101000010010001011001011000110100011001e-24793"},
{139, 4,
"1.020302230021023320300300101212330121100031233000032101123133120221012000000000000000000000000000000000000000000000000000000000000000000000e11221",
"1.001000110010101100001001001011111000110000110000010001100110111100011001010000001101101111000000001110010001011011011111011000101001000110e22442"},
{126, 13,
"4.a3cb351c6c548a0475218519514c6c54366681447019ac70a387862c39c86546ab27608c9c2863328860aa2464288070a76c0773882728c5213a335289259@2897",
"1.01011010000001110001100001101111100111011010010111000011000101111011000100001010010100110111101001001001000000011100010000000e10722"},
{6, 26,
"1.j79f6@-1593",
"1.00000e-7487"},
{26, 18,
"3.5e99682hh310aa89hb2fb4h88@-5704",
"1.0110010100010101000101100e-23784"},
{12, 21,
"4.j7f3e2ccdfa@-3524",
"1.10110101011e-15477"},
{38, 28,
"o.agr0m367b9bmm76rplg7b53qlj7f02g717cab@6452",
"1.1001010011101100110100111000111010001e31021"},
{75, 17,
"4.00abd9gc99902e1cae2caa7647gcc029g01370e96d3f8e9g02f814d3ge5faa29d40b9db470@5487",
"1.11100000110101010111101001110001001010111111010100000100001010100111011101e22429"},
{91, 16,
"1.0a812a627160014a3bda1f00000000000000000000000000000000000000000000000000000000000000000000@7897",
"1.000010101000000100101010011000100111000101100000000000010100101000111011110110100001111100e31588"},
{154, 19,
"1.989279dda02a8ic15e936ahig3c695f6059a3i01b7d1ge6a418bf84gd87e36061hb2bi62ciagcgb9226fafea41d2ig1e2f0a10ea3i40d6dahf598bdbh372bdf5901gh276866804ah53b6338bi@5285",
"1.110101101101101111110010001011110001100000010100011101101001000100110100000011110111000011011101011110010100110101011011111100101101001100000101101000010e22450"},
{53, 2,
"1.0100010111100111001010000100011011111011011100110111e-20319",
"1.0100010111100111001010000100011011111011011100110111e-20319"},
{76, 3,
"2.101212121100222100012112101120011222102000021110201110000202111122221100001e1511",
"1.000110101010111000011001011111110000001001101001011011111110111111010000111e2396"},
{31, 9,
"1.171774371505084376877631528681e3258",
"1.110101101011111011111000110011e10327"},
{175, 8,
"4.506242760242070533035566017365410474451421355546570157251400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e3483",
"1.001010001100101000101111100000101000100001110001010110110000111011011101100000011110111101011000010001001111001001010011000100010111011011011001101011110000011011110101010011e10451"},
{103, 24,
"8.0hmlm3g183cj358fn4bimn5bie1l89k95m647474mm8jg5kh1c011gi0m7de9j7b48c595g1bki4n32kll7b882eg7klgga0h0gf11@4510",
"1.001000110101001101011010101001111010110100010100110101010101110000001011001101110110010111000101010111e20681"},
{12, 9,
"3.00221080453e2479",
"1.11000111010e7859"},
{86, 11,
"6.873680186953174a274754118026423965415553a088387303452447389287133a0956111602a5a085446@5035",
"1.0000000000110100010110000111010001010100101011000100101010010011101010000110011110001e17421"},
{68, 10,
"6.1617378719016284192718392572980535262609909598793237475124371481233e481",
"1.0110001011000101110010111101100101111110001100001011110011001101111e1600"},
{11, 15,
"5.ab10c18d45@907",
"1.0000101111e3546"},
{77, 26,
"6.e6kl84g6h30o3nfnj7fjjff4n1ee6e5iop76gabj23e7hgan9o6724domc7bp4hdll95g817519f@5114",
"1.1011000101111111111110011011101100000100101000001001100000001011010001001000e24040"},
{28, 27,
"d.odiqp9kgh84o8d2aoqg4c21hemi@3566",
"1.101001111001111111110011110e16959"},
{45, 14,
"7.cddc6295a576980adbc8c16111d6301bad3146a1143c@-6227",
"1.10000000110011000000101100110001011100010111e-23706"},
{54, 19,
"1.b6e67i2124hfga2g819g1d6527g2b603eg3cd8hhca9gecig8geg1@4248",
"1.11010100100010101101110110010100000010111010010101110e18045"},
{49, 20,
"1.jj68bj6idadg44figi10d2ji99g6ddi6c6ich96a5h86i529@-3149",
"1.001011111101100100001010001000011100000000101110e-13609"},
{171, 16,
"6.22cf0e566d8ff11359d70bd9200065cfd72600b12e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@5602",
"1.10001000101100111100001110010101100110110110001111111100010001001101011001110101110000101111011001001000000000000001100101110011111101011100100110000000001011000100101110e22410"},
{144, 14,
"1.425d9709b4c125651ab88bb1a0370c14270d067a9a74a612dad48d5c025531c175c1b905201d0d9773aa686c8249db9c0b841b10821791c02baa2525a4aa7571850439c2cc965cd@-3351",
"1.11100111110001001101010111010000101010011000111001101011000001011110101110011011100100111001101101111011001001101011001101001011011101101111011e-12759"},
{166, 6,
"3.324252232403440543134003140400220120040245215204322153511143504542403430152410543444455151104314552352030352125540101550151410414122051500201022252511512332523431554e8340",
"1.010101111101111101001001110010111110010000001111010101100110011011010110011001001100001111010101100000010111011111101110110111101110010001110001111000001010001111000e21560"},
{141, 24,
"2.i3c88lkm2958l9ncb9f85kk35namjli84clek5j6jjkli82kb9m4e4i2g39me63db2094cif80gcba8ie6l15ia0d667kn9i1f77bdak599e1ach0j05cdn8kf6c6kfd82j2k6hj2c4d@4281",
"1.10011100001010110111001000000000101011100010101011001010001101110100110111011000111101000001111101100000110100100010101011001100100011001011e19629"},
{84, 6,
"2.41022133512503223022555143021524424430350133500020112434301542311050052304150111243e982",
"1.11010001111111001010011100011000011100100111111010001111010010101001001000011100001e2539"},
{56, 9,
"1.5305472255016741401411184703518332515066156086511016413e2936",
"1.0111110010001101000000110101110000110101001011001100111e9307"},
{18, 8,
"3.63542400000000000e-599",
"1.11100111011000101e-1796"},
{111, 13,
"8.b693ac7a24679b98708a0057a6202c867bc146740ab1971b380756a24c99804b63436419239ba0510030b819933771a636c57c5747b883@-6160",
"1.01011011111110100101110010100100000110000011011101001110010110000011101110111111010111000011011101101001100100e-22792"},
{162, 16,
"4.f2abe958a313566adbf3169e55cdcff3785dbd5c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@382",
"1.00111100101010101111101001010110001010001100010011010101100110101011011011111100110001011010011110010101011100110111001111111100110111100001011101101111010101110e1530"},
{117, 23,
"2.4b6kk3ag3if217ih1hggkk69bmcecfil1cd38dijh35j8e6ckhd335a4gj7l05bedk19473i8449b1ajc3jd3ka95eceheh72lh2jh17jamlm1142gll@-3628",
"1.10010010001010001110011000010000011111011101111100110101100100101111101110010011101001111010100010001111110100101111e-16411"},
{179, 2,
"1.1101101011111010101000110101010101101110001011011010101001110111011010011110001000000110101100010010001110010110011000000110001011111001011110100011101000110001001000110100100110e14203",
"1.1101101011111010101000110101010101101110001011011010101001110111011010011110001000000110101100010010001110010110011000000110001011111001011110100011101000110001001000110100100110e14203"},
{18, 27,
"4.ll743n2f654gh3154@-6039",
"1.01101001111010011e-28713"},
{178, 15,
"1.e5443105cad2d014b700c42aa3de854c4b95322420695d07db3564ec07473da83bde123b74c794139265a838ebeca745ad3dc97d7c356271ca935ea8e83306562c2a8edc6e886c1b6b2d3e17038379c33826526770985c068@821",
"1.011100001000101100111111111111000100110111110011101010001111011001111101111001010011100100100101100011101001000000101001010100011111001011001010011101101001000111111010101101011e3208"},
{161, 22,
"2.46ikji624bg042877h8g2jdki4ece6ede62841j7li843a4becdkkii86c54192jkefehikkb3kcb26ij1b3k9agfbb07dih88d6ej0ee0d63i8hedc7f0g0i9g7jf9gf6423j70h421bg5hf2bja9j0a432lb10@-5125",
"1.0111011000111110000010011100001100100110001011101001011110111010100000011100000010011101011100101100111100110000001101010101011110100011101111001011001111100000e-22854"},
{62, 19,
"7.bgd1g0886a6c3a9ee67cc7g3bgf718i98d90788idi5587358e660iffc0ic6@3257",
"1.0101100100001110000100010110100100000111110001111001011110100e13838"},
{127, 19,
"1.413bgf99eidba75ged25f7187080bce3h7ebdeghea4ig6c79g94di7b42a3e4cdi4ic6a53i71d2e4hdbe50ih0a0egf2fi469732131ig6g496bf7h8g3c86ie7h@-4465",
"1.001101111000011011100010010010010110111001001001110011110101111111000001110101111110001110010000110011111101000011000101111101e-18967"},
{17, 21,
"4.7d5b70gh4k0gj4fj@-116",
"1.1000100010000110e-508"},
{141, 13,
"2.2b4988c5cb57072a6a1a9c42224794a1cbc175a9bc673bb28aa045c3182b9396ca8bb8590969672b0239608a845a2c35c08908a58c2a83748c89241a6561422c7cc4866c8454@4358",
"1.10010110101000001000001001111001000100111110100010100110111011111011010010101000110101110000111100010000101101000110000000000001111110110011e16127"},
{39, 7,
"3.00350342452505221136410100232265245244e202",
"1.10011000111110011010100110101101010010e568"},
{119, 24,
"5.2aene587kc2d9a55mm8clhn4dn0a551de58b1fcli8e8hf1jlm7i0376dl5fhb2k8acka03077mnbn9d4dmi0641dce871c81g2b3ge76m3kngm4a9g5gh@-892",
"1.0111101010010100001001111110000000100101110010010111111100100101100001010010100110111000101100101010111000101111000010e-4088"},
{41, 14,
"5.c3dc5c49373d0c0075624931133022185bd08b16@-5294",
"1.0101011000010111111111000010100110011111e-20154"},
{41, 6,
"3.2411143454422033245255450304104450302500e2250",
"1.1110111101010101001001100000100011110111e5817"},
{17, 13,
"3.65789aa26aa273b1@-4490",
"1.1100011101010111e-16614"},
{10, 26,
"1.5p4hag1fl@6017",
"1.110010111e28282"},
{130, 11,
"2.606a72601843700427667823172635a47055021a0a68a99326875195a179483948407aa13726244552332114a1784aaa7239956521604460876871a65708458aa@-6285",
"1.110001001110111110110111000010101000110010011110010101100100001000101011010010000001000101000110111111110101000100000111100010100e-21742"},
{29, 20,
"j.4356d9b7i38i955jjj1j442501bj@163",
"1.1010101011110011100000100100e708"},
{140, 21,
"9.2f5k7aid6bj2b2g5bff29i73hk3a8d8g0i7ifa07hkb79g4hd3c7j6g4hjj2jbhai01gkje3h9g3gj3i34f0194kaed32iea9dcgcj8h7i1khdkf965c1ak97gf3h03fcab3ggi03fa@4864",
"1.0101011100011101000110101001010011111111010011000111111111100000011011100111010001100101100110001110001001100101001100110000011110100101101e21367"},
{133, 13,
"2.3721a9107307a71c75c07c83b70a25a9853619030b5bcb55101ca5c2060bca46c331b92b33aa957c3ac7c817335287c6917999c38c3806b6b5919623023ac52063bb@6602",
"1.011001101111100001100100110100010100010011100010111110110100100000000010011101001011000100000110011011101001010010011110111100010010e24431"},
{118, 2,
"1.001010111011011000100010001110111000001100101000101101010001110110000111101110111011011101111100110010000101001001001e18960",
"1.001010111011011000100010001110111000001100101000101101010001110110000111101110111011011101111100110010000101001001001e18960"},
{102, 23,
"l.26lhk42clcm9g940eihakhi32gb3331lld488cf1j4f73ge051bfl8gcmcg78gkjc2iibjf752eag0dee6dafa97k79jlh11j3270@-2160",
"1.01101011011000100101110111110001011000101101011001011111001101000110111010000010011111101110101100010e-9767"},
{156, 18,
"b.eb927dd4g48abee3cc2begehb9c3b8h83cae152db850ac2f3g816d6787825122c8h3aa3g8023h23000a8hg61065b3e367ac59ca373067730f96dd0d3b73b3c43fef91750b333gd497b8ce9228e7@5504",
"1.11000110111100011101100011001001110011101100011111010100101110010010010011111001100000011010011111111011001011111010001001011001110001100001101000000110000e22954"},
{158, 5,
"3.0112043214104344433122444210142004032004444213123303302023242414000243311324332203224340334422234000104132020124210141322013240010134130441413233111204013422e-10468",
"1.1001011000111111110100100101110011100001110100101001101110011001101001101011010010111010111111101010100011100010101100110111011101000110110100000111001100011e-24305"},
{7, 9,
"2.141540e-146",
"1.001111e-462"},
{111, 5,
"3.21233234110011204313222402442032333320324004133424222041314021020411320312421014434003440431230413141402230403e7641",
"1.10010000000101010000101010101011011010000100010010010000010110001111000111111111000110111001100101101110101101e17743"},
{76, 13,
"7.1c0861453a4ac156b6119ba7548251b5cb00b7c409c2bb8138214676468c9949676226013c1@4639",
"1.001000011000000011101101101010100010010001010111100110010101111110110010111e17169"},
{6, 25,
"c.aj660@-6978",
"1.11000e-32402"},
{156, 3,
"2.22101000022222000012110122210202211110020121210120112102122121111210000211020001020201202200011021211102012110220222110022001121011022011202000110120021012e-14744",
"1.11010001111000101111110000010011001101000100010010110011100100110001100111011101011111111100011111001100001111100101100000001000001100000000010010001011101e-23368"},
{7, 23,
"1.4hclk2@2148",
"1.110110e9716"},
{69, 11,
"2.77684920493191632416690544493465617a187218365952a6740034288687745a26@3263",
"1.01111000111000001111001110000110000110001111110011101100101111011100e11289"},
{146, 21,
"3.agg4d0dj636d526d4i8643ch5jee4ge2c3i46k121857dbedagd98cjifaf0fgc09ca739g2fkfbfh06i687kic2kb8c7i48gda57bb6d9bh81eh49h0d8e3i7ad2kgb1ek86b86g3589k27d@3562",
"1.0010111111111100101010101010001100110101010011011100001110111000101101001110001110010100000001010001000111010000010011110100010010101100101000001e15647"},
{20, 3,
"1.2000000021102111102e-16642",
"1.1011101011111110000e-26377"},
{68, 13,
"1.a43205b2164676727806614acc0398925569c3962a3ba419881a5c63b651aa3ab46@-618",
"1.1111011000001110010100111000110010110110011001110001100101011111000e-2287"},
{129, 4,
"2.22033002012102010122130132103000303000120122313322000222121000300000000000000000000000000000000000000000000000000000000000000000e13222",
"1.01010001111000010000110010010000100011010011100011110010011000000110011000000011000011010110111111010000000101010011001000000110e26445"},
{22, 6,
"1.420033001013011530142e11704",
"1.001000110010110110001e30255"},
{108, 6,
"1.03345424443433104422104400512453214240453335230205104304115343030341144544051005432030344054100542125304500e7375",
"1.00101101110001011101101111000010101011101000001111001110001101100000111100010101010101101100011110111010000e19064"},
{91, 27,
"2.ao077kf8oqoihn5pm6f5eqdcgnd2132d7p6n7di8ep82a1a9be99pm36g1emacbenaeiqphpgpdjhmm9ke3pn4pdea@-5482",
"1.111101100001000011101010001000000111000100100111110010101101110001101101101101101010111110e-26066"},
{96, 9,
"8.25805186310703506315505842015248775712246416686874260383701323213202658278523870037877823670166e-8134",
"1.11010111111000011100111001011010001110010001011101011101110101000101100100100010110011001010000e-25782"},
{161, 16,
"7.3a7627c1e42ef738698e292f0b81728c4b14fe8c000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@-3342",
"1.1100111010011101100010011111000001111001000010111011110111001110000110100110001110001010010010111100001011100000010111001010001100010010110001010011111110100011e-13366"},
{90, 3,
"2.10212200011211012002002221112120210222002020100202111000211012122020011102022112222021001e-3447",
"1.11100010111011011000101111110001000101000111110001100001010111101101011011110001000010001e-5463"},
{100, 27,
"a.f81hjjakdnc021op6ffh530ec8ige6n2fqc8f8j7ia7qelebgqkm4ic5ohh652hq1kgpag6pp0ldin6ce1fg6mj34077f5qc5oe@6576",
"1.011101001010010011110001100011111111010001110110100100101001010000101011101011110010010011111100000e31271"},
{152, 16,
"e.37ac48a0303f903c9d20883eddea4300d1190000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000@-1388",
"1.1100011011110101100010010001010000000110000001111111001000000111100100111010010000010001000001111101101110111101010010000110000000011010001000110010000e-5549"},
{106, 20,
"1.3g2h7i2776d50gjibi937f8cdci3idecdeh3j2gba0j8d1ghgg3eg609ji55h5g7jeai1bii3a4f9jhjfij6jd1g3cg0f6024e252gc3e@6422",
"1.100110000101011010100111100110000000100101000110110011010010000101000100110010001110011110111100010000111e27755"},
{23, 17,
"9.f72e724454d1g0f60g93g6@-6563",
"1.0011100011110110010001e-26823"},
{98, 6,
"1.2444223304453415235424343034030405514010421403514410005234221430055051205523402412055242134042045e-8535",
"1.1101110011010001101001001111100101111010100111001011110001000010100101101110011011101100000111011e-22063"},
{4, 18,
"1.gec@-6711",
"1.100e-27984"},
{69, 24,
"8.d45gdfnhkhb7a20nj96dnggic83imhjne0cceldechn1m4e9fbd9db0ablngjf9n7810@6975",
"1.00100111111100101100110011110110110000110110110010100101011111000100e31983"},
{122, 8,
"4.0227760456667717717077553523466457265600000000000000000000000000000000000000000000000000000000000000000000000000000000000e-1767",
"1.0000001001011111111000010010111011011011111100111111100111100011111110110101110101001110011011010010111101011010111000000e-5299"},
{144, 23,
"8.b01c48dg20bek9a5k376clc501aecg92bdjaeji2dm9230i7j3k36jm50b0c5a0753i2b18534cji34bcl2li033cc534m52k2gbegc25a5g30lf4calag58026i5d7li61jg9digj5ceb1@-4456",
"1.00010000110011010111011011110111001101111001010110001101011100100101101110110000010011011111100000100110001001001111111011010110000000001111110e-20154"},
{111, 4,
"2.23100111310122202021232133233012212012232222323230133100000000000000000000000000000000000000000000000000000000e-10458",
"1.01011010000010101110100011010100010001001101110011111101111000110100110000110101110101010111011101100011111010e-20915"},
{117, 10,
"1.61207328685870427963864999744776917701013812304254540861834226053316419217753608451422967376154318603744156166920074e-6440",
"1.01100011000100111001100010000000110010100001001011111010100001101111100100101100111010100011101110001010011010010010e-21393"},
{106, 16,
"1.dd30a1d24091263243ca1c144f0000000000000000000000000000000000000000000000000000000000000000000000000000000@354",
"1.110111010011000010100001110100100100000010010001001001100011001001000011110010100001110000010100010011110e1416"},
{77, 14,
"4.90d6913ba57b149d8d85a58c311b4d537c10bd7d3c10d69c62bc08d32269760126a58115a105@-7311",
"1.1001000000111100000111001001011000110101001111100001100111010100010000011111e-27834"},
{8, 4,
"3.2230000e15197",
"1.1101011e30395"},
{81, 24,
"1.84ni25h558abmhg2dk7bl2jbbmkf4i8i2bemc5cgmk9jf301c00k24271m9h7mgm4301be1lnldn4364@2573",
"1.01110010011000110110100101011001011111101111101100010110101101011101100001000010e11797"},
{94, 2,
"1.010010010101111001001011111111100100011110110100010001101111111100100101101100110101001011111e32427",
"1.010010010101111001001011111111100100011110110100010001101111111100100101101100110101001011111e32427"},
{77, 21,
"1.87e4k9df85g50ead6fcj4h86820ikdjdjg93d90ca406g470hhkh7ciaba1aggg753g36553ebh5@2538",
"1.0010001100011000111010000010011001010011000000100101010001100000111101000111e11148"},
{80, 17,
"1.923gga999230g94fce02efg753ce001045a35e0264c9c2cb17850e32484fc3526dcg38ed874g5f2@3392",
"1.0011100111101001001101111001110100001100111110011110110001100110101010111001110e13865"},
{99, 7,
"4.53646362336126606650465342500160461331562113222506144210636341332436342025203333264316511653025011e-5540",
"1.01101101111001001100001101101101010011001001100110111000010000101000011001001001101000011101011001e-15551"},
{119, 20,
"1.c8966iabcf4de94ad15f9e83j407i3he7fch54h5jh0g5d74e06c057gg72a107didj8d1j8geibbfec5j36c3fgd5e12edjb9g10j7c9i03f33hi80ce0@7153",
"1.0101110101100011110001001110100110011000100000001001000110111110011111100011111010011101011111101101010011110111110100e30915"},
{93, 13,
"2.c749cb562c3a758b1a21a650666a4c6c53c76ca58a1a75a0358c9ac3866887972b3551a03aa6c150856531258508@193",
"1.10101111101001011010111101100100111110011111010110111101100100010011001001100011110100111110e715"},
{145, 14,
"1.c61614b64261d22c62cb9d16163ca4d144ac23351b708506b3b610b1b67b764ca974448d7a2c6515a6bc97503d4b2a530c75b2b677a464c6629c69b6c3d7860d7749b4b653c434d5@2050",
"1.111111100001101111100011001111100010010000101000011110000001110100111001011010100001001010111111010001111101000110011000011101110110001001100101e7805"},
{159, 23,
"4.bj9l07l0215e7l6lf1dkf62i056l37jaa0gdih717656f1kk1a77883jf99jg31le43em76bmcg4lddl782ihkla0m392886d8lm67d6c3a1l4j12kg0l1k52ee77lmk0gech11g8jeei680k85bi460c7el17@-1539",
"1.01010100110100100101100001011100000001100011110001001101000010000001000010000110000110010001110100001101011101101001001101101111001101101111101001010010010100e-6960"},
{24, 25,
"g.m749al09kflg5b42jnn4a7b@-2820",
"1.01010010101011010111011e-13092"},
{88, 18,
"3.5ed0gad0bhhb7aa9ge2ad1dhcg6833f3e068936hghf23gd2aa69f13539f15hfce50aa64achfee49bfg7249g@-4058",
"1.001000010110011011000101100000101111101001100011101101001111110111000010010110010001100e-16920"}
};

static void
check_reftable (void)
{
  int i, base;
  mpfr_t x, y;
  mpfr_prec_t p;
  char *s;

  mpfr_init2 (x, 200);
  mpfr_init2 (y, 200);
  for (i = 0 ; i < numberof (RefTable) ; i++)
    {
      base = RefTable[i].base;
      p    = RefTable[i].prec;
      mpfr_set_prec (x, p);
      mpfr_set_prec (y, p);
      mpfr_set_str_binary (x, RefTable[i].binstr);
      mpfr_strtofr (y, RefTable[i].str, &s, base, MPFR_RNDN);
      if (s == NULL || *s != 0)
        {
          printf ("strtofr didn't parse entire input for i=%d:\n"
                  " Str=%s", i, RefTable[i].str);
          exit (1);
        }
      if (mpfr_cmp (x, y))
        {
          printf ("Results differ between strtofr and set_binary for i=%d:\n"
                  " Set binary gives: ", i);
          mpfr_dump (x);
          printf (" strtofr    gives: ");
          mpfr_dump (y);
          printf (" setstr     gives: ");
          mpfr_set_str (x, RefTable[i].str, base, MPFR_RNDN);
          mpfr_dump (x);
          mpfr_set_prec (x, 2*p);
          mpfr_set_str (x, RefTable[i].str, base, MPFR_RNDN);
          printf (" setstr ++  gives: ");
          mpfr_dump (x);
          exit (1);
        }
    }
  mpfr_clear (y);
  mpfr_clear (x);
}

static void
check_parse (void)
{
  mpfr_t x;
  char *s;
  int res;

  mpfr_init (x);

  /* Invalid data */
  mpfr_set_si (x, -1, MPFR_RNDN);
  res = mpfr_strtofr (x, "  invalid", NULL, 10, MPFR_RNDN);
  if (MPFR_NOTZERO (x) || MPFR_IS_NEG (x))
    {
      printf ("Failed parsing '  invalid' (1)\n X=");
      mpfr_dump (x);
      exit (1);
    }
  MPFR_ASSERTN (res == 0);
  mpfr_set_si (x, -1, MPFR_RNDN);
  res = mpfr_strtofr (x, "  invalid", &s, 0, MPFR_RNDN);
  if (MPFR_NOTZERO (x) || MPFR_IS_NEG (x) || strcmp (s, "  invalid"))
    {
      printf ("Failed parsing '  invalid' (2)\n S=%s\n X=", s);
      mpfr_dump (x);
      exit (1);
    }
  MPFR_ASSERTN (res == 0);
  /* Check if it stops correctly */
  mpfr_strtofr (x, "15*x", &s, 10, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 15) || strcmp (s, "*x"))
    {
      printf ("Failed parsing '15*x'\n S=%s\n X=", s);
      mpfr_dump (x);
      exit (1);
    }
  /* Check for leading spaces */
  mpfr_strtofr (x, "  1.5E-10 *x^2", &s, 10, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "1.5E-10") || strcmp (s, " *x^2"))
    {
      printf ("Failed parsing '1.5E-10*x^2'\n S=%s\n X=", s);
      mpfr_dump (x);
      exit (1);
    }
  /* Check for leading sign */
  mpfr_strtofr (x, "  +17.5E-42E ", &s, 10, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "17.5E-42") || strcmp (s, "E "))
    {
      printf ("Failed parsing '+17.5E-42E '\n S=%s\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-17.5E+42E\n", &s, 10, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "-17.5E42") || strcmp (s, "E\n"))
    {
      printf ("Failed parsing '-17.5E+42\\n'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* P form */
  mpfr_strtofr (x, "0x42P17", &s, 16, MPFR_RNDN);
  if (mpfr_cmp_str (x, "8650752", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '0x42P17' (base = 16)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-0X42p17", &s, 16, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-8650752", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '-0x42p17' (base = 16)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "42p17", &s, 16, MPFR_RNDN);
  if (mpfr_cmp_str (x, "8650752", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '42p17' (base = 16)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-42P17", &s, 16, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-8650752", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '-42P17' (base = 16)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "0b1001P17", &s, 2, MPFR_RNDN);
  if (mpfr_cmp_str (x, "1179648", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '0b1001P17' (base = 2)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-0B1001p17", &s, 2, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-1179648", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '-0B1001p17' (base = 2)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "1001p17", &s, 2, MPFR_RNDN);
  if (mpfr_cmp_str (x, "1179648", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '1001p17' (base = 2)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-1001P17", &s, 2, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-1179648", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '-1001P17' (base = 2)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* Check for auto-detection of the base */
  mpfr_strtofr (x, "+0x42P17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "42P17", 16, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '+0x42P17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-42E17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-42E17", 10, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '-42E17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-42P17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "-42", 10, MPFR_RNDN) || strcmp (s, "P17"))
    {
      printf ("Failed parsing '-42P17' (base = 0)\n S='%s'\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, " 0b0101.011@42", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0101.011@42", 2, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '0101.011@42'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, " 0b0101.011P42", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "0101.011@42", 2, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '0101.011@42'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "+0x42@17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str (x, "4.2@18", 16, MPFR_RNDN) || *s != 0)
    {
      printf ("Failed parsing '+0x42P17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }


  /* Check for space inside the mantissa */
  mpfr_strtofr (x, "+0x4 2@17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 4) || strcmp(s," 2@17"))
    {
      printf ("Failed parsing '+0x4 2@17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "+0x42 P17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0x42) || strcmp(s," P17"))
    {
      printf ("Failed parsing '+0x42 P17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* Space between mantissa and exponent */
  mpfr_strtofr (x, " -0b0101P 17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_si (x, -5) || strcmp(s,"P 17"))
    {
      printf ("Failed parsing '-0b0101P 17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* Check for Invalid exponent. */
  mpfr_strtofr (x, " -0b0101PF17", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_si (x, -5) || strcmp(s,"PF17"))
    {
      printf ("Failed parsing '-0b0101PF17'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* At least one digit in the mantissa. */
  mpfr_strtofr (x, " .E10", &s, 0, MPFR_RNDN);
  if (strcmp(s," .E10"))
    {
      printf ("Failed parsing ' .E10'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* Check 2 '.': 2.3.4   */
  mpfr_strtofr (x, "-1.2.3E4", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_str1 (x, "-1.2") || strcmp(s,".3E4"))
    {
      printf ("Failed parsing '-1.2.3E4'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 10, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  /* Check for 0x and 0b */
  mpfr_strtofr (x, "  0xG", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) || strcmp(s,"xG"))
    {
      printf ("Failed parsing '  0xG'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "  0b2", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) || strcmp(s,"b2"))
    {
      printf ("Failed parsing '  0b2'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 2, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "-0x.23@2Z33", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_si (x, -0x23) || strcmp(s,"Z33"))
    {
      printf ("Failed parsing '-0x.23@2Z33'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }
  mpfr_strtofr (x, "  0x", &s, 0, MPFR_RNDN);
  if (mpfr_cmp_ui (x, 0) || strcmp(s,"x"))
    {
      printf ("Failed parsing '  0x'\n S=%s\n X=", s);
      mpfr_out_str (stdout, 16, 0, x, MPFR_RNDN); putchar ('\n');
      exit (1);
    }

  mpfr_clear (x);
}

static void
check_overflow (void)
{
  mpfr_t x;
  char *s;

  mpfr_init (x);

  /* Huge overflow */
  mpfr_strtofr (x, "123456789E2147483646", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check overflow failed (1) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "123456789E9223372036854775807", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check overflow failed (2) with:\n s='%s'\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "123456789E170141183460469231731687303715884105728",
                &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check overflow failed (3) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }

  /* Limit overflow */
  mpfr_strtofr (x, "12E2147483646", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check overflow failed (4) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "12E2147483645", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x))
    {
      printf ("Check overflow failed (5) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "0123456789ABCDEF@2147483640", &s, 16, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x))
    {
      printf ("Check overflow failed (6) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "0123456789ABCDEF@540000000", &s, 16, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_INF (x) || !MPFR_IS_POS (x))
    {
      printf ("Check overflow failed (7) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }

  /* Check underflow */
  mpfr_strtofr (x, "123456789E-2147483646", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_ZERO (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check underflow failed (1) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "123456789E-9223372036854775807", &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_ZERO (x) || !MPFR_IS_POS (x) )
    {
      printf ("Check underflow failed (2) with:\n s='%s'\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "-123456789E-170141183460469231731687303715884105728",
                &s, 0, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_ZERO (x) || !MPFR_IS_NEG (x) )
    {
      printf ("Check underflow failed (3) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }
  mpfr_strtofr (x, "0123456789ABCDEF@-540000000", &s, 16, MPFR_RNDN);
  if (s[0] != 0 || !MPFR_IS_ZERO (x) || !MPFR_IS_POS (x))
    {
      printf ("Check overflow failed (7) with:\n s=%s\n x=", s);
      mpfr_dump (x);
      exit (1);
    }

  mpfr_clear (x);
}

static void
check_retval (void)
{
  mpfr_t x;
  int res;

  mpfr_init2 (x, 10);

  res = mpfr_strtofr (x, "01011000111", NULL, 2, MPFR_RNDN);
  MPFR_ASSERTN (res == 0);
  res = mpfr_strtofr (x, "11011000111", NULL, 2, MPFR_RNDN);
  MPFR_ASSERTN (res > 0);
  res = mpfr_strtofr (x, "110110001101", NULL, 2, MPFR_RNDN);
  MPFR_ASSERTN (res < 0);

  mpfr_clear (x);
}

/* Bug found by Christoph Lauter (in mpfr_set_str). */
static struct bug20081025_test {
  mpfr_rnd_t rnd;
  int inexact;
  const char *str;
  const char *binstr;
} Bug20081028Table[] = {
  {MPFR_RNDN, -1, "1.00000000000000000006", "1"},
  {MPFR_RNDZ, -1, "1.00000000000000000006", "1"},
  {MPFR_RNDU, +1, "1.00000000000000000006",
   "10000000000000000000000000000001e-31"},
  {MPFR_RNDD, -1, "1.00000000000000000006", "1"},


  {MPFR_RNDN, +1, "-1.00000000000000000006", "-1"},
  {MPFR_RNDZ, +1, "-1.00000000000000000006", "-1"},
  {MPFR_RNDU, +1, "-1.00000000000000000006", "-1"},
  {MPFR_RNDD, -1, "-1.00000000000000000006",
   "-10000000000000000000000000000001e-31"},

  {MPFR_RNDN, +1, "0.999999999999999999999999999999999999999999999", "1"},
  {MPFR_RNDZ, -1, "0.999999999999999999999999999999999999999999999",
   "11111111111111111111111111111111e-32"},
  {MPFR_RNDU, +1, "0.999999999999999999999999999999999999999999999", "1"},
  {MPFR_RNDD, -1, "0.999999999999999999999999999999999999999999999",
   "11111111111111111111111111111111e-32"},

  {MPFR_RNDN, -1, "-0.999999999999999999999999999999999999999999999", "-1"},
  {MPFR_RNDZ, +1, "-0.999999999999999999999999999999999999999999999",
   "-11111111111111111111111111111111e-32"},
  {MPFR_RNDU, +1, "-0.999999999999999999999999999999999999999999999",
   "-11111111111111111111111111111111e-32"},
  {MPFR_RNDD, -1, "-0.999999999999999999999999999999999999999999999", "-1"}
};

static void
bug20081028 (void)
{
  int i;
  int inexact, res;
  mpfr_rnd_t rnd;
  mpfr_t x, y;
  char *s;

  mpfr_init2 (x, 32);
  mpfr_init2 (y, 32);
  for (i = 0 ; i < numberof (Bug20081028Table) ; i++)
    {
      rnd     = Bug20081028Table[i].rnd;
      inexact = Bug20081028Table[i].inexact;
      mpfr_set_str_binary (x, Bug20081028Table[i].binstr);
      res = mpfr_strtofr (y, Bug20081028Table[i].str, &s, 10, rnd);
      if (s == NULL || *s != 0)
        {
          printf ("Error in Bug20081028: strtofr didn't parse entire input\n"
                  "for (i=%d) Str=\"%s\"", i, Bug20081028Table[i].str);
          exit (1);
        }
      if (! SAME_SIGN (res, inexact))
        {
          printf ("Error in Bug20081028: expected %s ternary value, "
                  "got %d\nfor (i=%d) Rnd=%s Str=\"%s\"\n Set binary gives: ",
                  inexact > 0 ? "positive" : "negative",
                  res, i, mpfr_print_rnd_mode(rnd), Bug20081028Table[i].str);
          mpfr_dump (x);
          printf (" strtofr    gives: ");
          mpfr_dump (y);
          exit (1);
        }
      if (mpfr_cmp (x, y))
        {
          printf ("Error in Bug20081028: Results differ between strtofr and "
                  "set_binary\nfor (i=%d) Rnd=%s Str=\"%s\"\n"
                  " Set binary gives: ",
                  i, mpfr_print_rnd_mode(rnd), Bug20081028Table[i].str);
          mpfr_dump (x);
          printf (" strtofr    gives: ");
          mpfr_dump (y);
          exit (1);
        }
    }
  mpfr_clear (y);
  mpfr_clear (x);
}

/* check that 1.23e is correctly parsed, cf
   http://gmplib.org/list-archives/gmp-bugs/2010-March/001898.html */
static void
test20100310 (void)
{
  mpfr_t x, y;
  char str[] = "1.23e", *endptr;

  mpfr_init2 (x, 53);
  mpfr_init2 (y, 53);
  mpfr_strtofr (x, str, &endptr, 10, MPFR_RNDN);
  mpfr_strtofr (y, "1.23", NULL, 10, MPFR_RNDN);
  if (mpfr_cmp (x, y) != 0)
    {
      printf ("x <> y in test20100310\n");
      exit (1);
    }
  if (endptr != str + 4) /* strtofr should take into account '1.23',
                            not '1.23e' */
    {
      printf ("endptr <> str + 4 in test20100310\n");
      exit (1);
    }
  mpfr_clear (x);
  mpfr_clear (y);
}

/* From a bug reported by Joseph S. Myers
   https://sympa.inria.fr/sympa/arc/mpfr/2012-08/msg00005.html */
static void
bug20120814 (void)
{
  mpfr_exp_t emin = -30, e;
  mpfr_t x, y;
  int r;
  char s[64], *p;

  mpfr_init2 (x, 2);
  mpfr_set_ui_2exp (x, 3, emin - 2, MPFR_RNDN);
  mpfr_get_str (s + 1, &e, 10, 19, x, MPFR_RNDD);
  s[0] = s[1];
  s[1] = '.';
  for (p = s; *p != 0; p++) ;
  *p = 'e';
  sprintf (p + 1, "%d", (int) e - 1);

  mpfr_init2 (y, 4);
  r = mpfr_strtofr (y, s, NULL, 0, MPFR_RNDN);
  if (r <= 0 || ! mpfr_equal_p (x, y))
    {
      printf ("Error in bug20120814\n");
      printf ("mpfr_strtofr failed on string \"%s\"\n", s);
      printf ("Expected inex > 0 and y = 0.1100E%d\n", (int) emin);
      printf ("Got inex = %-6d and y = ", r);
      mpfr_dump (y);
      exit (1);
    }

  mpfr_clear (x);
  mpfr_clear (y);
}

static void
bug20120829 (void)
{
  mpfr_t x1, x2, e;
  int inex1, inex2, i, r;
  char s[48] = "1e-1";

  mpfr_init2 (e, 128);
  mpfr_inits2 (4, x1, x2, (mpfr_ptr) 0);

  inex1 = mpfr_set_si (e, -1, MPFR_RNDN);
  MPFR_ASSERTN (inex1 == 0);

  for (i = 1; i <= sizeof(s) - 5; i++)
    {
      s[3+i] = '0';
      s[4+i] = 0;
      inex1 = mpfr_mul_ui (e, e, 10, MPFR_RNDN);
      MPFR_ASSERTN (inex1 == 0);
      RND_LOOP(r)
        {
          mpfr_rnd_t rnd = (mpfr_rnd_t) r;

          inex1 = mpfr_exp10 (x1, e, rnd);
          inex1 = SIGN (inex1);
          inex2 = mpfr_strtofr (x2, s, NULL, 0, rnd);
          inex2 = SIGN (inex2);
          /* On 32-bit machines, for i = 7, r8389, r8391 and r8394 do:
             strtofr.c:...: MPFR assertion failed: cy == 0
             r8396 is OK.
             On 64-bit machines, for i = 15,
             r8389 does: strtofr.c:678: MPFR assertion failed: err < (64 - 0)
             r8391 does: strtofr.c:680: MPFR assertion failed: h < ysize
             r8394 and r8396 are OK.
          */
          if (! mpfr_equal_p (x1, x2) || inex1 != inex2)
            {
              printf ("Error in bug20120829 for i = %d, rnd = %s\n",
                      i, mpfr_print_rnd_mode (rnd));
              printf ("Expected inex = %d, x = ", inex1);
              mpfr_dump (x1);
              printf ("Got      inex = %d, x = ", inex2);
              mpfr_dump (x2);
              exit (1);
            }
        }
    }

  mpfr_clears (e, x1, x2, (mpfr_ptr) 0);
}

/* Note: the number is 5^47/2^9. */
static void
bug20161217 (void)
{
  mpfr_t fp, z;
  static const char * num = "0.1387778780781445675529539585113525390625e31";
  int inex;

  mpfr_init2 (fp, 110);
  mpfr_init2 (z, 110);
  inex = mpfr_strtofr (fp, num, NULL, 10, MPFR_RNDN);
  MPFR_ASSERTN(inex == 0);
  mpfr_set_str_binary (z, "10001100001000010011110110011101101001010000001011011110010001010100010100100110111101000010001011001100001101E-9");
  MPFR_ASSERTN(mpfr_equal_p (fp, z));
  mpfr_clear (fp);
  mpfr_clear (z);
}

int
main (int argc, char *argv[])
{
  tests_start_mpfr ();

  check_special();
  check_reftable ();
  check_parse ();
  check_overflow ();
  check_retval ();
  bug20081028 ();
  test20100310 ();
  bug20120814 ();
  bug20120829 ();
  bug20161217 ();

  tests_end_mpfr ();
  return 0;
}
