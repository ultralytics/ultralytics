/* Prototypes etc for calc program.

Copyright 2001 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see https://www.gnu.org/licenses/.  */

#include <stddef.h>  /* for size_t */
#ifndef NO_CALC_H
#include "calc.h"
#endif
#include "calc-config.h"

struct calc_keywords_t {
  char  *name;
  int   value;
};

extern int  calc_option_readline;
extern int  calc_more_input;
extern const struct calc_keywords_t  calc_keywords[];

int calc_input (char *buf, size_t max_size);
void calc_init_readline (void);
