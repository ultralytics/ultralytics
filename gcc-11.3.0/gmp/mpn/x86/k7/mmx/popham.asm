dnl  AMD K7 mpn_popcount, mpn_hamdist -- population count and hamming
dnl  distance.

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


C			     popcount	     hamdist
C P3 generic			6.5		7
C P3 model 9  (Banias)          5.7		6.1
C P3 model 13 (Dothan)		5.75		6
C K7				5		6

C unsigned long mpn_popcount (mp_srcptr src, mp_size_t size);
C unsigned long mpn_hamdist (mp_srcptr src, mp_srcptr src2, mp_size_t size);
C
C The code here is almost certainly not optimal, but is already a 3x speedup
C over the generic C code.  The main improvement would be to interleave
C processing of two qwords in the loop so as to fully exploit the available
C execution units, possibly leading to 3.25 c/l (13 cycles for 4 limbs).
C
C The loop is based on the example "Efficient 64-bit population count using
C MMX instructions" in the Athlon Optimization Guide, AMD document 22007,
C page 158 of rev E (reference in mpn/x86/k7/README).

ifdef(`OPERATION_popcount',,
`ifdef(`OPERATION_hamdist',,
`m4_error(`Need OPERATION_popcount or OPERATION_hamdist defined
')')')

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
')

	TEXT
	ALIGN(32)

PROLOGUE(M4_function)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx

ifdef(`PIC',`
	movl	$0xAAAAAAAA, %eax
	movl	$0x33333333, %edx

	movd	%eax, %mm7
	movd	%edx, %mm6

	movl	$0x0F0F0F0F, %eax

	punpckldq %mm7, %mm7
	punpckldq %mm6, %mm6

	movd	%eax, %mm5
	movd	%edx, %mm4

	punpckldq %mm5, %mm5

',`
	movq	L(rodata_AAAAAAAAAAAAAAAA), %mm7
	movq	L(rodata_3333333333333333), %mm6
	movq	L(rodata_0F0F0F0F0F0F0F0F), %mm5
')
	pxor	%mm4, %mm4

define(REG_AAAAAAAAAAAAAAAA,%mm7)
define(REG_3333333333333333,%mm6)
define(REG_0F0F0F0F0F0F0F0F,%mm5)
define(REG_0000000000000000,%mm4)


	movl	PARAM_SRC, %eax
HAM(`	movl	PARAM_SRC2, %edx')

	pxor	%mm2, %mm2	C total

	shrl	%ecx
	jnc	L(top)

	movd	(%eax,%ecx,8), %mm1

HAM(`	movd	(%edx,%ecx,8), %mm0
	pxor	%mm0, %mm1
')
	orl	%ecx, %ecx
	jmp	L(loaded)


	ALIGN(16)
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
	decl	%ecx

L(loaded):
	movq	%mm1, %mm0
	pand	REG_AAAAAAAAAAAAAAAA, %mm1

	psrlq	$1, %mm1

	psubd	%mm1, %mm0	C bit pairs


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


	psadbw(	%mm4, %mm0)

	paddd	%mm0, %mm2	C add to total
	jnz	L(top)


	movd	%mm2, %eax
	emms
	ret

EPILOGUE()
