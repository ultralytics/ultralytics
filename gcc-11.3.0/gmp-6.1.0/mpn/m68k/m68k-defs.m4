divert(-1)

dnl  m4 macros for 68k assembler.

dnl  Copyright 2001-2003 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.


dnl  The default m4 `#' commenting interferes with the assembler syntax for
dnl  immediates.  `|' would be correct, but it interferes with "||" in
dnl  eval().  Would like to disable commenting, but that's not possible (see
dnl  mpn/asm-defs.m4), so use `;' which should be harmless.

changecom(;)


dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo)
dnl
dnl  Same as the standard PROLOGUE, but align to 2 bytes not 4.

define(`PROLOGUE_cpu',
m4_assert_numargs(1)
`	TEXT
	ALIGN(2)
	GLOBL	`$1' GLOBL_ATTR
	TYPE(`$1',`function')
`$1'LABEL_SUFFIX')


dnl  Usage: d0, etc
dnl
dnl  Expand to d0 or %d0 according to the assembler's requirements.
dnl
dnl  Actually d0 expands to `d0' or %`d0', the quotes protecting against
dnl  further expansion.  Definitions are made even if d0 is to be just `d0',
dnl  so that any m4 quoting problems will show up everywhere, not just on a
dnl  %d0 system.
dnl
dnl  Care must be taken with quoting when using these in a definition.  For
dnl  instance the quotes in the following are essential or two %'s will be
dnl  produced when `counter' is used.
dnl
dnl         define(counter, `d7')
dnl

dnl  Called: m68k_reg(r)
define(m68k_reg,
m4_assert_numargs(1)
m4_assert_defined(`WANT_REGISTER_PERCENT')
`ifelse(WANT_REGISTER_PERCENT,yes,%)`$1'')

dnl  Usage: m68k_defreg(r)
define(m68k_defreg,
m4_assert_numargs(1)
`deflit($1,`m68k_reg(`$1')')')

m68k_defreg(d0)
m68k_defreg(d1)
m68k_defreg(d2)
m68k_defreg(d3)
m68k_defreg(d4)
m68k_defreg(d5)
m68k_defreg(d6)
m68k_defreg(d7)

m68k_defreg(a0)
m68k_defreg(a1)
m68k_defreg(a2)
m68k_defreg(a3)
m68k_defreg(a4)
m68k_defreg(a5)
m68k_defreg(a6)
m68k_defreg(a7)

m68k_defreg(sp)
m68k_defreg(pc)


dnl  Usage: M(base)
dnl         M(base,displacement)
dnl         M(base,index,size)
dnl         M(base,index,size,scale)
dnl         M(base,+)
dnl         M(-,base)
dnl
dnl  `base' is an address register, `index' is a data register, `size' is w
dnl  or l, and scale is 1, 2, 4 or 8.
dnl
dnl  M(-,base) has it's arguments that way around to emphasise it's a
dnl  pre-decrement, as opposed to M(base,+) a post-increment.
dnl
dnl  Enhancement: Add the memory indirect modes, if/when they're needed.

define(M,
m4_assert_numargs_range(1,4)
m4_assert_defined(`WANT_ADDRESSING')
`ifelse(WANT_ADDRESSING,mit,
`ifelse($#,1, ``$1'@')dnl
ifelse($#,2,
`ifelse($2,+, ``$1'@+',
`ifelse($1,-, ``$2'@-',
              ``$1'@($2)')')')dnl
ifelse($#,3,  ``$1'@(`$2':`$3')')dnl
ifelse($#,4,  ``$1'@(`$2':`$3':$4)')',

dnl  WANT_ADDRESSING `motorola'
`ifelse($#,1, `(`$1')')dnl
ifelse($#,2,
`ifelse($2,+, `(`$1')+',
`ifelse($1,-, `-(`$2')',
              `$2(`$1')')')')dnl
ifelse($#,3,  `(`$1',`$2'.$3)')dnl
ifelse($#,4,  `(`$1',`$2'.$3*$4)')')')


dnl  Usage: addl etc
dnl
dnl  m68k instructions with special handling for the suffix, with for
dnl  instance addl expanding to addl or add.l as necessary.
dnl
dnl  See also t-m68k-defs.pl which verifies all mnemonics used in the asm
dnl  files have entries here.

dnl  Called: m68k_insn(mnemonic,suffix)
define(m68k_insn,
m4_assert_numargs(2)
m4_assert_defined(`WANT_DOT_SIZE')
`ifelse(WANT_DOT_SIZE,yes, ``$1'.``$2''',
                           ``$1$2'')')

dnl  Usage: m68k_definsn(mnemonic,suffix)
define(m68k_definsn,
m4_assert_numargs(2)
`deflit($1`'$2,`m68k_insn(`$1',`$2')')')

m68k_definsn(add,  l)
m68k_definsn(addx, l)
m68k_definsn(addq, l)
m68k_definsn(asl,  l)
m68k_definsn(cmp,  l)
m68k_definsn(cmp,  w)
m68k_definsn(clr,  l)
m68k_definsn(divu, l)
m68k_definsn(eor,  w)
m68k_definsn(lsl,  l)
m68k_definsn(lsr,  l)
m68k_definsn(move, l)
m68k_definsn(move, w)
m68k_definsn(movem,l)
m68k_definsn(moveq,l)
m68k_definsn(mulu, l)
m68k_definsn(neg,  l)
m68k_definsn(or,   l)
m68k_definsn(roxl, l)
m68k_definsn(roxr, l)
m68k_definsn(sub,  l)
m68k_definsn(subx, l)
m68k_definsn(subq, l)


dnl  Usage: bra etc
dnl
dnl  Expand to `bra', `jra' or `jbra' according to what the assembler will
dnl  accept.  The latter two give variable-sized branches in gas.
dnl
dnl  See also t-m68k-defs.pl which verifies all the bXX branches used in the
dnl  asm files have entries here.

dnl  Called: m68k_branch(cond)
define(m68k_branch,
m4_assert_numargs(1)
m4_assert_defined(`WANT_BRANCHES')
`ifelse(WANT_BRANCHES,jra, `j$1',
`ifelse(WANT_BRANCHES,jbra,`jb$1',
                           ``b$1'')')')

dnl  Called: m68k_defbranch(cond)
define(m68k_defbranch,
m4_assert_numargs(1)
`deflit(b$1,`m68k_branch(`$1')')')

m68k_defbranch(ra)
m68k_defbranch(cc)
m68k_defbranch(cs)
m68k_defbranch(ls)
m68k_defbranch(eq)
m68k_defbranch(ne)


dnl  Usage: scale_available_p
dnl
dnl  Expand to 1 if a scale factor can be used in addressing modes, or 0 if
dnl  not.  M(a0,d0,l,4), meaning a0+d0*4, is not available in 68000 or
dnl  68010, but is in CPU32 and in 68020 and up.

define(scale_available_p,
`m4_ifdef_anyof_p(
`HAVE_HOST_CPU_m68360'
`HAVE_HOST_CPU_m68020'
`HAVE_HOST_CPU_m68030'
`HAVE_HOST_CPU_m68040'
`HAVE_HOST_CPU_m68060')')


divert
