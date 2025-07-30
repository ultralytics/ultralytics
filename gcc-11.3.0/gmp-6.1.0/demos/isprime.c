/* Classify numbers as probable primes, primes or composites.
   With -q return true if the following argument is a (probable) prime.

Copyright 1999, 2000, 2002, 2005, 2012 Free Software Foundation, Inc.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see https://www.gnu.org/licenses/.  */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "gmp.h"

char *progname;

void
print_usage_and_exit ()
{
  fprintf (stderr, "usage: %s -q nnn\n", progname);
  fprintf (stderr, "usage: %s nnn ...\n", progname);
  exit (-1);
}

int
main (int argc, char **argv)
{
  mpz_t n;
  int i;

  progname = argv[0];

  if (argc < 2)
    print_usage_and_exit ();

  mpz_init (n);

  if (argc == 3 && strcmp (argv[1], "-q") == 0)
    {
      if (mpz_set_str (n, argv[2], 0) != 0)
	print_usage_and_exit ();
      exit (mpz_probab_prime_p (n, 25) == 0);
    }

  for (i = 1; i < argc; i++)
    {
      int class;
      if (mpz_set_str (n, argv[i], 0) != 0)
	print_usage_and_exit ();
      class = mpz_probab_prime_p (n, 25);
      mpz_out_str (stdout, 10, n);
      if (class == 0)
	puts (" is composite");
      else if (class == 1)
	puts (" is a probable prime");
      else /* class == 2 */
	puts (" is a prime");
    }
  exit (0);
}
