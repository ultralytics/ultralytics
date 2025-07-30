/* Lexical analyzer for calc program.

Copyright 2000-2002 Free Software Foundation, Inc.

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

%{
#include <string.h>
#include "calc-common.h"


#if WITH_READLINE
/* Let GNU flex use readline.  See the calcread.c redefined input() for a
   way that might work for a standard lex too.  */
#define YY_INPUT(buf,result,max_size)   \
  result = calc_input (buf, max_size);
#endif


/* Non-zero when reading the second or subsequent line of an expression,
   used to give a different prompt when using readline.  */
int  calc_more_input = 0;


const struct calc_keywords_t  calc_keywords[] = {
  { "abs",       ABS },
  { "bin",       BIN },
  { "decimal",   DECIMAL },
  { "fib",       FIB },
  { "hex",       HEX },
  { "help",      HELP },
  { "gcd",       GCD },
  { "kron",      KRON },
  { "lcm",       LCM },
  { "lucnum",    LUCNUM },
  { "nextprime", NEXTPRIME },
  { "powm",      POWM },
  { "quit",      QUIT },
  { "root",      ROOT },
  { "sqrt",      SQRT },
  { NULL }
};
%}

%%

[ \t\f] { /* white space is skipped */ }

[;\n]   { /* semicolon or newline separates statements */
          calc_more_input = 0;
          return EOS; }
\\\n    { /* escaped newlines are skipped */ }


#(([^\\\n]*)\\)+\n {
            /* comment through to escaped newline is skipped */ }
#[^\n]*\n { /* comment through to newline is a separator */
            calc_more_input = 0;
            return EOS; }
#[^\n]* {   /* comment through to EOF skipped */ }


[-+*/%()<>^!=,] { return yytext[0]; }
"<="    { return LE; }
">="    { return GE; }
"=="    { return EQ; }
"!="    { return NE; }
"<<"    { return LSHIFT; }
">>"    { return RSHIFT; }
"&&"    { return LAND; }
"||"    { return LOR; }

(0[xX])?[0-9A-F]+ {
        yylval.str = yytext;
        return NUMBER; }

[a-zA-Z][a-zA-Z0-9]* {
        int  i;

        for (i = 0; calc_keywords[i].name != NULL; i++)
          if (strcmp (yytext, calc_keywords[i].name) == 0)
            return calc_keywords[i].value;

        if (yytext[0] >= 'a' && yytext[0] <= 'z' && yytext[1] == '\0')
          {
            yylval.var = yytext[0] - 'a';
            return VARIABLE;
          }

        return BAD;
}

. { return BAD; }

%%

int
yywrap ()
{
  return 1;
}
