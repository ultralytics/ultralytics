/* x86 fat binary initializers.

   THE FUNCTIONS AND VARIABLES IN THIS FILE ARE FOR INTERNAL USE ONLY.
   THEY'RE ALMOST CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR
   COMPLETELY IN FUTURE GNU MP RELEASES.

Copyright 2003, 2004, 2011-2013, 2015 Free Software Foundation, Inc.

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

#include <stdio.h>    /* for printf */
#include <stdlib.h>   /* for getenv */
#include <string.h>

#include "gmp.h"
#include "gmp-impl.h"

/* Change this to "#define TRACE(x) x" for some traces. */
#define TRACE(x)


/* fat_entry.asm */
long __gmpn_cpuid (char [12], int);
int  __gmpn_cpuid_available (void);


#if WANT_FAKE_CPUID
/* The "name"s in the table are values for the GMP_CPU_TYPE environment
   variable.  Anything can be used, but for now it's the canonical cpu types
   as per config.guess/config.sub.  */

#define __gmpn_cpuid            fake_cpuid
#define __gmpn_cpuid_available  fake_cpuid_available

#define MAKE_FMS(family, model)						\
  ((((family) & 0xf) << 8) + (((family) & 0xff0) << 20)			\
   + (((model) & 0xf) << 4) + (((model)  &  0xf0) << 12))

static struct {
  const char  *name;
  const char  *vendor;
  unsigned    fms;
} fake_cpuid_table[] = {
  { "i386",       "" },
  { "i486",       "GenuineIntel", MAKE_FMS (4, 0) },
  { "pentium",    "GenuineIntel", MAKE_FMS (5, 0) },
  { "pentiummmx", "GenuineIntel", MAKE_FMS (5, 4) },
  { "pentiumpro", "GenuineIntel", MAKE_FMS (6, 0) },
  { "pentium2",   "GenuineIntel", MAKE_FMS (6, 2) },
  { "pentium3",   "GenuineIntel", MAKE_FMS (6, 7) },
  { "pentium4",   "GenuineIntel", MAKE_FMS (15, 2) },
  { "prescott",   "GenuineIntel", MAKE_FMS (15, 3) },
  { "nocona",     "GenuineIntel", MAKE_FMS (15, 4) },
  { "core2",      "GenuineIntel", MAKE_FMS (6, 0xf) },
  { "nehalem",    "GenuineIntel", MAKE_FMS (6, 0x1a) },
  { "nhm",        "GenuineIntel", MAKE_FMS (6, 0x1a) },
  { "atom",       "GenuineIntel", MAKE_FMS (6, 0x1c) },
  { "westmere",   "GenuineIntel", MAKE_FMS (6, 0x25) },
  { "wsm",        "GenuineIntel", MAKE_FMS (6, 0x25) },
  { "sandybridge","GenuineIntel", MAKE_FMS (6, 0x2a) },
  { "sbr",        "GenuineIntel", MAKE_FMS (6, 0x2a) },
  { "silvermont", "GenuineIntel", MAKE_FMS (6, 0x37) },
  { "slm",        "GenuineIntel", MAKE_FMS (6, 0x37) },
  { "haswell",    "GenuineIntel", MAKE_FMS (6, 0x3c) },
  { "hwl",        "GenuineIntel", MAKE_FMS (6, 0x3c) },
  { "broadwell",  "GenuineIntel", MAKE_FMS (6, 0x3d) },
  { "bwl",        "GenuineIntel", MAKE_FMS (6, 0x3d) },
  { "skylake",    "GenuineIntel", MAKE_FMS (6, 0x5e) },
  { "sky",        "GenuineIntel", MAKE_FMS (6, 0x5e) },

  { "k5",         "AuthenticAMD", MAKE_FMS (5, 0) },
  { "k6",         "AuthenticAMD", MAKE_FMS (5, 3) },
  { "k62",        "AuthenticAMD", MAKE_FMS (5, 8) },
  { "k63",        "AuthenticAMD", MAKE_FMS (5, 9) },
  { "athlon",     "AuthenticAMD", MAKE_FMS (6, 0) },
  { "k8",         "AuthenticAMD", MAKE_FMS (15, 0) },
  { "k10",        "AuthenticAMD", MAKE_FMS (16, 0) },
  { "bobcat",     "AuthenticAMD", MAKE_FMS (20, 1) },
  { "bulldozer",  "AuthenticAMD", MAKE_FMS (21, 1) },
  { "piledriver", "AuthenticAMD", MAKE_FMS (21, 2) },
  { "steamroller","AuthenticAMD", MAKE_FMS (21, 0x30) },
  { "excavator",  "AuthenticAMD", MAKE_FMS (21, 0x60) },
  { "jaguar",     "AuthenticAMD", MAKE_FMS (22, 1) },

  { "viac3",      "CentaurHauls", MAKE_FMS (6, 0) },
  { "viac32",     "CentaurHauls", MAKE_FMS (6, 9) },
  { "nano",       "CentaurHauls", MAKE_FMS (6, 15) },
};

static int
fake_cpuid_lookup (void)
{
  char  *s;
  int   i;

  s = getenv ("GMP_CPU_TYPE");
  if (s == NULL)
    {
      printf ("Need GMP_CPU_TYPE environment variable for fake cpuid\n");
      abort ();
    }

  for (i = 0; i < numberof (fake_cpuid_table); i++)
    if (strcmp (s, fake_cpuid_table[i].name) == 0)
      return i;

  printf ("GMP_CPU_TYPE=%s unknown\n", s);
  abort ();
}

static int
fake_cpuid_available (void)
{
  return fake_cpuid_table[fake_cpuid_lookup()].vendor[0] != '\0';
}

static long
fake_cpuid (char dst[12], int id)
{
  int  i = fake_cpuid_lookup();

  switch (id) {
  case 0:
    memcpy (dst, fake_cpuid_table[i].vendor, 12);
    return 0;
  case 1:
    return fake_cpuid_table[i].fms;
  default:
    printf ("fake_cpuid(): oops, unknown id %d\n", id);
    abort ();
  }
}
#endif


typedef DECL_preinv_divrem_1 ((*preinv_divrem_1_t));
typedef DECL_preinv_mod_1    ((*preinv_mod_1_t));

struct cpuvec_t __gmpn_cpuvec = {
  __MPN(add_n_init),
  0,
  0,
  __MPN(addmul_1_init),
  0,
  __MPN(bdiv_dbm1c_init),
  __MPN(cnd_add_n_init),
  __MPN(cnd_sub_n_init),
  __MPN(com_init),
  __MPN(copyd_init),
  __MPN(copyi_init),
  __MPN(divexact_1_init),
  __MPN(divrem_1_init),
  __MPN(gcd_1_init),
  __MPN(lshift_init),
  __MPN(lshiftc_init),
  __MPN(mod_1_init),
  __MPN(mod_1_1p_init),
  __MPN(mod_1_1p_cps_init),
  __MPN(mod_1s_2p_init),
  __MPN(mod_1s_2p_cps_init),
  __MPN(mod_1s_4p_init),
  __MPN(mod_1s_4p_cps_init),
  __MPN(mod_34lsub1_init),
  __MPN(modexact_1c_odd_init),
  __MPN(mul_1_init),
  __MPN(mul_basecase_init),
  __MPN(mullo_basecase_init),
  __MPN(preinv_divrem_1_init),
  __MPN(preinv_mod_1_init),
  __MPN(redc_1_init),
  __MPN(redc_2_init),
  __MPN(rshift_init),
  __MPN(sqr_basecase_init),
  __MPN(sub_n_init),
  0,
  __MPN(submul_1_init),
  0
};

int __gmpn_cpuvec_initialized = 0;

/* The following setups start with generic x86, then overwrite with
   specifics for a chip, and higher versions of that chip.

   The arrangement of the setups here will normally be the same as the $path
   selections in configure.in for the respective chips.

   This code is reentrant and thread safe.  We always calculate the same
   decided_cpuvec, so if two copies of the code are running it doesn't
   matter which completes first, both write the same to __gmpn_cpuvec.

   We need to go via decided_cpuvec because if one thread has completed
   __gmpn_cpuvec then it may be making use of the threshold values in that
   vector.  If another thread is still running __gmpn_cpuvec_init then we
   don't want it to write different values to those fields since some of the
   asm routines only operate correctly up to their own defined threshold,
   not an arbitrary value.  */

void
__gmpn_cpuvec_init (void)
{
  struct cpuvec_t  decided_cpuvec;

  TRACE (printf ("__gmpn_cpuvec_init:\n"));

  memset (&decided_cpuvec, '\0', sizeof (decided_cpuvec));

  CPUVEC_SETUP_x86;
  CPUVEC_SETUP_fat;

  if (! __gmpn_cpuid_available ())
    {
      TRACE (printf ("  80386, or early 80486 without cpuid\n"));
    }
  else
    {
      char vendor_string[13];
      char dummy_string[12];
      long fms;
      int family, model;

      __gmpn_cpuid (vendor_string, 0);
      vendor_string[12] = 0;

      fms = __gmpn_cpuid (dummy_string, 1);
      family = ((fms >> 8) & 0xf) + ((fms >> 20) & 0xff);
      model = ((fms >> 4) & 0xf) + ((fms >> 12) & 0xf0);

      if (strcmp (vendor_string, "GenuineIntel") == 0)
        {
          switch (family)
            {
            case 4:
              TRACE (printf ("  80486 with cpuid\n"));
              break;

            case 5:
              TRACE (printf ("  pentium\n"));
              CPUVEC_SETUP_pentium;
              if (model >= 4)
                {
                  TRACE (printf ("  pentiummmx\n"));
                  CPUVEC_SETUP_pentium_mmx;
                }
              break;

            case 6:
              TRACE (printf ("  p6\n"));
              CPUVEC_SETUP_p6;
	      switch (model)
		{
		case 0x00:
		case 0x01:
		  TRACE (printf ("  pentiumpro\n"));
		  break;

		case 0x02:
		case 0x03:
		case 0x04:
		case 0x05:
		case 0x06:
		  TRACE (printf ("  pentium2\n"));
                  CPUVEC_SETUP_p6_mmx;
		  break;

		case 0x07:
		case 0x08:
		case 0x0a:
		case 0x0b:
		case 0x0c:
		  TRACE (printf ("  pentium3\n"));
                  CPUVEC_SETUP_p6_mmx;
                  CPUVEC_SETUP_p6_p3mmx;
		  break;

		case 0x09:		/* Banias */
		case 0x0d:		/* Dothan */
		case 0x0e:		/* Yonah */
		  TRACE (printf ("  Banias/Dothan/Yonah\n"));
                  CPUVEC_SETUP_p6_mmx;
                  CPUVEC_SETUP_p6_p3mmx;
                  CPUVEC_SETUP_p6_sse2;
		  break;

		case 0x0f:		/* Conroe Merom Kentsfield Allendale */
		case 0x10:
		case 0x11:
		case 0x12:
		case 0x13:
		case 0x14:
		case 0x15:
		case 0x16:
		case 0x17:		/* PNR Wolfdale Yorkfield */
		case 0x18:
		case 0x19:
		case 0x1d:		/* PNR Dunnington */
		  TRACE (printf ("  Conroe\n"));
                  CPUVEC_SETUP_p6_mmx;
                  CPUVEC_SETUP_p6_p3mmx;
                  CPUVEC_SETUP_p6_sse2;
		  CPUVEC_SETUP_core2;
		  break;

		case 0x1c:		/* Atom Silverthorne */
		case 0x26:		/* Atom Lincroft */
		case 0x27:		/* Atom Saltwell */
		case 0x36:		/* Atom Cedarview/Saltwell */
		  TRACE (printf ("  atom\n"));
		  CPUVEC_SETUP_atom;
		  CPUVEC_SETUP_atom_mmx;
		  CPUVEC_SETUP_atom_sse2;
		  break;

		case 0x1a:		/* NHM Gainestown */
		case 0x1b:
		case 0x1e:		/* NHM Lynnfield/Jasper */
		case 0x1f:
		case 0x20:
		case 0x21:
		case 0x22:
		case 0x23:
		case 0x24:
		case 0x25:		/* WSM Clarkdale/Arrandale */
		case 0x28:
		case 0x29:
		case 0x2b:
		case 0x2c:		/* WSM Gulftown */
		case 0x2e:		/* NHM Beckton */
		case 0x2f:		/* WSM Eagleton */
		  TRACE (printf ("  nehalem/westmere\n"));
                  CPUVEC_SETUP_p6_mmx;
                  CPUVEC_SETUP_p6_p3mmx;
                  CPUVEC_SETUP_p6_sse2;
		  CPUVEC_SETUP_core2;
		  CPUVEC_SETUP_coreinhm;
		  break;

		case 0x2a:		/* SBR */
		case 0x2d:		/* SBR-EP */
		case 0x3a:		/* IBR */
		case 0x3c:		/* Haswell */
		  TRACE (printf ("  sandybridge\n"));
                  CPUVEC_SETUP_p6_mmx;
                  CPUVEC_SETUP_p6_p3mmx;
                  CPUVEC_SETUP_p6_sse2;
		  CPUVEC_SETUP_core2;
		  CPUVEC_SETUP_coreinhm;
		  CPUVEC_SETUP_coreisbr;
		  break;
		}
              break;

            case 15:
              TRACE (printf ("  pentium4\n"));
              CPUVEC_SETUP_pentium4;
              CPUVEC_SETUP_pentium4_mmx;
              CPUVEC_SETUP_pentium4_sse2;
              break;
            }
        }
      else if (strcmp (vendor_string, "AuthenticAMD") == 0)
        {
          switch (family)
            {
            case 5:
              if (model <= 3)
                {
                  TRACE (printf ("  k5\n"));
                }
              else
                {
                  TRACE (printf ("  k6\n"));
                  CPUVEC_SETUP_k6;
                  CPUVEC_SETUP_k6_mmx;
                  if (model >= 8)
                    {
                      TRACE (printf ("  k62\n"));
                      CPUVEC_SETUP_k6_k62mmx;
                    }
                  if (model >= 9)
                    {
                      TRACE (printf ("  k63\n"));
                    }
                }
              break;
            case 6:
              TRACE (printf ("  athlon\n"));
              CPUVEC_SETUP_k7;
              CPUVEC_SETUP_k7_mmx;
              break;

            case 0x0f:		/* k8 */
            case 0x11:		/* "fam 11h", mix of k8 and k10 */
            case 0x13:		/* unknown, conservatively assume k8  */
            case 0x16:		/* unknown, conservatively assume k8  */
            case 0x17:		/* unknown, conservatively assume k8  */
              TRACE (printf ("  k8\n"));
              CPUVEC_SETUP_k7;
              CPUVEC_SETUP_k7_mmx;
              CPUVEC_SETUP_k8;
	      break;

            case 0x10:		/* k10 */
            case 0x12:		/* k10 (llano) */
              TRACE (printf ("  k10\n"));
              CPUVEC_SETUP_k7;
              CPUVEC_SETUP_k7_mmx;
	      break;

            case 0x14:		/* bobcat */
              TRACE (printf ("  bobcat\n"));
              CPUVEC_SETUP_k7;
              CPUVEC_SETUP_k7_mmx;
              CPUVEC_SETUP_bobcat;
	      break;

            case 0x15:		/* bulldozer */
              TRACE (printf ("  bulldozer\n"));
              CPUVEC_SETUP_k7;
              CPUVEC_SETUP_k7_mmx;
	      break;
            }
        }
      else if (strcmp (vendor_string, "CentaurHauls") == 0)
        {
          switch (family)
            {
            case 6:
              TRACE (printf ("  viac3\n"));
              if (model >= 9)
                {
                  TRACE (printf ("  viac32\n"));
                }
	      if (model >= 15)
		{
                  TRACE (printf ("  nano\n"));
		  CPUVEC_SETUP_nano;
		}
              break;
            }
        }
      else if (strcmp (vendor_string, "CyrixInstead") == 0)
        {
          /* Should recognize Cyrix' processors too.  */
          TRACE (printf ("  cyrix something\n"));
        }
    }

  /* There's no x86 generic mpn_preinv_divrem_1 or mpn_preinv_mod_1.
     Instead default to the plain versions from whichever CPU we detected.
     The function arguments are compatible, no need for any glue code.  */
  if (decided_cpuvec.preinv_divrem_1 == NULL)
    decided_cpuvec.preinv_divrem_1 =(preinv_divrem_1_t)decided_cpuvec.divrem_1;
  if (decided_cpuvec.preinv_mod_1 == NULL)
    decided_cpuvec.preinv_mod_1    =(preinv_mod_1_t)   decided_cpuvec.mod_1;

  ASSERT_CPUVEC (decided_cpuvec);
  CPUVEC_INSTALL (decided_cpuvec);

  /* Set this once the threshold fields are ready.
     Use volatile to prevent it getting moved.  */
  *((volatile int *) &__gmpn_cpuvec_initialized) = 1;
}
