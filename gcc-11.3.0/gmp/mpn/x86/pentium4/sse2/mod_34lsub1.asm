dnl  Intel Pentium 4 mpn_mod_34lsub1 -- remainder modulo 2^24-1.

dnl  Copyright 2000-2003 Free Software Foundation, Inc.

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


C Pentium4: 1.0 cycles/limb


C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C
C Enhancements:
C
C There might a couple of cycles to save by using plain integer code for
C more small sizes.  2 limbs measures about 20 cycles, but 3 limbs jumps to
C about 46 (inclusive of some function call overheads).

defframe(PARAM_SIZE, 8)
defframe(PARAM_SRC,  4)

dnl  re-use parameter space
define(SAVE_EBX, `PARAM_SRC')
define(SAVE_ESI, `PARAM_SIZE')

	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_34lsub1)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %edx
	movl	(%edx), %eax

	subl	$2, %ecx
	ja	L(three_or_more)
	jne	L(one)

	movl	4(%edx), %edx
	movl	%eax, %ecx
	shrl	$24, %eax		C src[0] high

	andl	$0x00FFFFFF, %ecx	C src[0] low
	addl	%ecx, %eax

	movl	%edx, %ecx
	shll	$8, %edx

	shrl	$16, %ecx		C src[1] low
	addl	%ecx, %eax

	andl	$0x00FFFF00, %edx	C src[1] high
	addl	%edx, %eax

L(one):
	ret


L(three_or_more):
	pxor	%mm0, %mm0
	pxor	%mm1, %mm1
	pxor	%mm2, %mm2

	pcmpeqd	%mm7, %mm7
	psrlq	$32, %mm7	C 0x00000000FFFFFFFF, low 32 bits

	pcmpeqd	%mm6, %mm6
	psrlq	$40, %mm6	C 0x0000000000FFFFFF, low 24 bits

L(top):
	C eax
	C ebx
	C ecx	counter, size-2 to 0, -1 or -2
	C edx	src, incrementing
	C
	C mm0	sum 0mod3
	C mm1	sum 1mod3
	C mm2	sum 2mod3
	C mm3
	C mm4
	C mm5
	C mm6	0x0000000000FFFFFF
	C mm7	0x00000000FFFFFFFF

	movd	(%edx), %mm3
	paddq	%mm3, %mm0

	movd	4(%edx), %mm3
	paddq	%mm3, %mm1

	movd	8(%edx), %mm3
	paddq	%mm3, %mm2

	addl	$12, %edx
	subl	$3, %ecx
	ja	L(top)


	C ecx is -2, -1 or 0 representing 0, 1 or 2 more limbs, respectively

	addl	$1, %ecx
	js	L(combine)		C 0 more

	movd	(%edx), %mm3
	paddq	%mm3, %mm0

	jz	L(combine)		C 1 more

	movd	4(%edx), %mm3
	paddq	%mm3, %mm1

L(combine):
	movq	%mm7, %mm3		C low halves
	pand	%mm0, %mm3

	movq	%mm7, %mm4
	pand	%mm1, %mm4

	movq	%mm7, %mm5
	pand	%mm2, %mm5

	psrlq	$32, %mm0		C high halves
	psrlq	$32, %mm1
	psrlq	$32, %mm2

	paddq	%mm0, %mm4		C fold high halves to give 33 bits each
	paddq	%mm1, %mm5
	paddq	%mm2, %mm3

	psllq	$8, %mm4		C combine at respective offsets
	psllq	$16, %mm5
	paddq	%mm4, %mm3
	paddq	%mm5, %mm3		C 0x000cxxxxxxxxxxxx, 50 bits

	pand	%mm3, %mm6		C fold at 24 bits
	psrlq	$24, %mm3

	paddq	%mm6, %mm3
	movd	%mm3, %eax

	ASSERT(z,	C nothing left in high dword
	`psrlq	$32, %mm3
	movd	%mm3, %ecx
	orl	%ecx, %ecx')

	emms
	ret

EPILOGUE()
