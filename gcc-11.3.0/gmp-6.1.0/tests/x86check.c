/* x86 calling conventions checking. */

/*
Copyright 2000, 2001, 2010 Free Software Foundation, Inc.

This file is part of the GNU MP Library test suite.

The GNU MP Library test suite is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 3 of the License,
or (at your option) any later version.

The GNU MP Library test suite is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the GNU MP Library test suite.  If not, see https://www.gnu.org/licenses/.  */

#include <stdio.h>
#include "gmp.h"
#include "gmp-impl.h"
#include "tests.h"


/* Vector if constants and register values.  We use one vector to allow access
   via a base pointer, very beneficial for the PIC-enabled amd64call.asm.  */
mp_limb_t calling_conventions_values[17] =
{
  CNST_LIMB(0x12345678),	/* want_ebx */
  CNST_LIMB(0x89ABCDEF),	/* want_ebp */
  CNST_LIMB(0xDEADBEEF),	/* want_esi */
  CNST_LIMB(0xFFEEDDCC),	/* want_edi */

  CNST_LIMB(0xFEEDABBA),	/* JUNK_EAX */
  CNST_LIMB(0xAB78DE89),	/* JUNK_ECX */
  CNST_LIMB(0x12389018)		/* JUNK_EDX */

  /* rest of array used for dynamic values.  */
};

/* Index starts for various regions in above vector.  */
#define WANT	0
#define JUNK	4
#define SAVE	7
#define RETADDR	11
#define VAL	12
#define EFLAGS	16


/* values to check */
#ifdef __cplusplus
extern "C" {
#endif
struct {
  unsigned  control;
  unsigned  status;
  unsigned  tag;
  unsigned  other[4];
} calling_conventions_fenv;
#ifdef __cplusplus
}
#endif

/* expected values, as per x86call.asm */
#define VALUE_EBX   0x01234567
#define VALUE_ESI   0x89ABCDEF
#define VALUE_EDI   0xFEDCBA98
#define VALUE_EBP   0x76543210


const char *regname[] = {"ebx", "ebp", "esi", "edi"};

#define DIR_BIT(eflags)   (((eflags) & (1<<10)) != 0)


/* Return 1 if ok, 0 if not */

int
calling_conventions_check (void)
{
  const char  *header = "Violated calling conventions:\n";
  int  ret = 1;
  int i;

#define CHECK(callreg, regstr, value)                   \
  if (callreg != value)                                 \
    {                                                   \
      printf ("%s   %s  got 0x%08X want 0x%08X\n",      \
              header, regstr, callreg, value);          \
      header = "";                                      \
      ret = 0;                                          \
    }

  for (i = 0; i < 4; i++)
    {
      CHECK (calling_conventions_values[VAL+i], regname[i], calling_conventions_values[WANT+i]);
    }

  if (DIR_BIT (calling_conventions_values[EFLAGS]) != 0)
    {
      printf ("%s   eflags dir bit  got %d want 0\n",
              header, DIR_BIT (calling_conventions_values[EFLAGS]));
      header = "";
      ret = 0;
    }

  if ((calling_conventions_fenv.tag & 0xFFFF) != 0xFFFF)
    {
      printf ("%s   fpu tags  got 0x%X want 0xFFFF\n",
              header, calling_conventions_fenv.tag & 0xFFFF);
      header = "";
      ret = 0;
    }

  return ret;
}
