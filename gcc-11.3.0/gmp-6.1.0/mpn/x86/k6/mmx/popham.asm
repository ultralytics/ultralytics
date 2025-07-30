dnl  AMD K6-2 mpn_popcount, mpn_hamdist -- mpn bit population count and
dnl  hamming distance.

dnl  Copyright 2000-2002 Free Software Foundation, Inc.

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

include(`../config.m4')


C        popcount  hamdist
C K6-2:    9.0       11.5   cycles/limb
C K6:      12.5      13.0


C unsigned long mpn_popcount (mp_srcptr src, mp_size_t size);
C unsigned long mpn_hamdist (mp_srcptr src, mp_srcptr src2, mp_size_t size);
C
C The code here isn't optimal, but it's already a 2x speedup over the plain
C integer mpn/generic/popcount.c,hamdist.c.


ifdef(`OPERATION_popcount',,
`ifdef(`OPERATION_hamdist',,
`m4_error(`Need OPERATION_popcount or OPERATION_hamdist
')m4exit(1)')')

define(HAM,
m4_assert_numargs(1)
`ifdef(`OPERATION_hamdist',`$1')')

define(POP,
m4_assert_numargs(1)
`ifdef(`OPERATION_popcount',`$1')')

HAM(`
defframe(PARAM_SIZE,   12)
defframe(PARAM_SRC2,   8)
defframe(PARAM_SRC,    4)
define(M4_function,mpn_hamdist)
')
POP(`
defframe(PARAM_SIZE,   8)
defframe(PARAM_SRC,    4)
define(M4_function,mpn_popcount)
')

MULFUNC_PROLOGUE(mpn_popcount mpn_hamdist)


ifdef(`PIC',,`
	dnl  non-PIC

	RODATA
	ALIGN(8)

L(rodata_AAAAAAAAAAAAAAAA):
	.long	0xAAAAAAAA
	.long	0xAAAAAAAA

L(rodata_3333333333333333):
	.long	0x33333333
	.long	0x33333333

L(rodata_0F0F0F0F0F0F0F0F):
	.long	0x0F0F0F0F
	.long	0x0F0F0F0F

L(rodata_000000FF000000FF):
	.long	0x000000FF
	.long	0x000000FF
')

	TEXT
	ALIGN(32)

POP(`ifdef(`PIC', `
	C avoid shrl crossing a 32-byte boundary
	nop')')

PROLOGUE(M4_function)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx

ifdef(`PIC',`
	movl	$0xAAAAAAAA, %eax
	movl	$0x33333333, %edx

	movd	%eax, %mm7
	movd	%edx, %mm6

	movl	$0x0F0F0F0F, %eax
	movl	$0x000000FF, %edx

	punpckldq %mm7, %mm7
	punpckldq %mm6, %mm6

	movd	%eax, %mm5
	movd	%edx, %mm4

	punpckldq %mm5, %mm5
	punpckldq %mm4, %mm4
',`

	movq	L(rodata_AAAAAAAAAAAAAAAA), %mm7
	movq	L(rodata_3333333333333333), %mm6
	movq	L(rodata_0F0F0F0F0F0F0F0F), %mm5
	movq	L(rodata_000000FF000000FF), %mm4
')

define(REG_AAAAAAAAAAAAAAAA, %mm7)
define(REG_3333333333333333, %mm6)
define(REG_0F0F0F0F0F0F0F0F, %mm5)
define(REG_000000FF000000FF, %mm4)


	movl	PARAM_SRC, %eax
HAM(`	movl	PARAM_SRC2, %edx')

	pxor	%mm2, %mm2	C total

	shrl	%ecx
	jnc	L(top)

Zdisp(	movd,	0,(%eax,%ecx,8), %mm1)

HAM(`
Zdisp(	movd,	0,(%edx,%ecx,8), %mm0)
	pxor	%mm0, %mm1
')

	incl	%ecx
	jmp	L(loaded)


	ALIGN(16)
POP(`	nop	C alignment to avoid crossing 32-byte boundaries')

L(top):
	C eax	src
	C ebx
	C ecx	counter, qwords, decrementing
	C edx	[hamdist] src2
	C
	C mm0	(scratch)
	C mm1	(scratch)
	C mm2	total (low dword)
	C mm3
	C mm4	\
	C mm5	| special constants
	C mm6	|
	C mm7	/

	movq	-8(%eax,%ecx,8), %mm1
HAM(`	pxor	-8(%edx,%ecx,8), %mm1')

L(loaded):
	movq	%mm1, %mm0
	pand	REG_AAAAAAAAAAAAAAAA, %mm1

	psrlq	$1, %mm1
HAM(`	nop			C code alignment')

	psubd	%mm1, %mm0	C bit pairs
HAM(`	nop			C code alignment')


	movq	%mm0, %mm1
	psrlq	$2, %mm0

	pand	REG_3333333333333333, %mm0
	pand	REG_3333333333333333, %mm1

	paddd	%mm1, %mm0	C nibbles


	movq	%mm0, %mm1
	psrlq	$4, %mm0

	pand	REG_0F0F0F0F0F0F0F0F, %mm0
	pand	REG_0F0F0F0F0F0F0F0F, %mm1

	paddd	%mm1, %mm0	C bytes

	movq	%mm0, %mm1
	psrlq	$8, %mm0


	paddb	%mm1, %mm0	C words


	movq	%mm0, %mm1
	psrlq	$16, %mm0

	paddd	%mm1, %mm0	C dwords

	pand	REG_000000FF000000FF, %mm0

	paddd	%mm0, %mm2	C low to total
	psrlq	$32, %mm0

	paddd	%mm0, %mm2	C high to total
	loop	L(top)



	movd	%mm2, %eax
	emms_or_femms
	ret

EPILOGUE()
