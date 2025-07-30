dnl  Intel Pentium-4 mpn_submul_1 -- Multiply a limb vector with a limb and
dnl  subtract the result from a second limb vector.

dnl  Copyright 2001, 2002, 2008, 2010 Free Software Foundation, Inc.

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


C			    cycles/limb
C P6 model 0-8,10-12		-
C P6 model 9   (Banias)		6.8
C P6 model 13  (Dothan)		6.9
C P4 model 0-1 (Willamette)	?
C P4 model 2   (Northwood)	5.87
C P4 model 3-4 (Prescott)	6.5

C This code represents a step forwards compared to the code available before
C GMP 5.1, but it is not carefully tuned for either P6 or P4.  In fact, it is
C not good for P6.  For P4 it saved a bit over 1 c/l for both Northwood and
C Prescott compared to the old code.
C
C The arrangements made here to get a two instruction dependent chain are
C slightly subtle.  In the loop the carry (or borrow rather) is a negative so
C that a paddq can be used to give a low limb ready to store, and a high limb
C ready to become the new carry after a psrlq.
C
C If the carry was a simple twos complement negative then the psrlq shift would
C need to bring in 0 bits or 1 bits according to whether the high was zero or
C non-zero, since a non-zero value would represent a negative needing sign
C extension.  That wouldn't be particularly easy to arrange and certainly would
C add an instruction to the dependent chain, so instead an offset is applied so
C that the high limb will be 0xFFFFFFFF+c.  With c in the range -0xFFFFFFFF to
C 0, the value 0xFFFFFFFF+c is in the range 0 to 0xFFFFFFFF and is therefore
C always positive and can always have 0 bits shifted in, which is what psrlq
C does.
C
C The extra 0xFFFFFFFF must be subtracted before c is used, but that can be
C done off the dependent chain.  The total adjustment then is to add
C 0xFFFFFFFF00000000 to offset the new carry, and subtract 0x00000000FFFFFFFF
C to remove the offset from the current carry, for a net add of
C 0xFFFFFFFE00000001.  In the code this is applied to the destination limb when
C fetched.
C
C It's also possible to view the 0xFFFFFFFF adjustment as a ones-complement
C negative, which is how it's undone for the return value, but that doesn't
C seem as clear.

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT
	ALIGN(16)

PROLOGUE(mpn_submul_1c)
deflit(`FRAME',0)
	movd	PARAM_CARRY, %mm1
	jmp	L(start_1c)
EPILOGUE()

PROLOGUE(mpn_submul_1)
deflit(`FRAME',0)
	pxor	%mm1, %mm1		C initial borrow

L(start_1c):
	mov	PARAM_SRC, %eax
	pcmpeqd	%mm0, %mm0

	movd	PARAM_MULTIPLIER, %mm7
	pcmpeqd	%mm6, %mm6

	mov	PARAM_DST, %edx
	psrlq	$32, %mm0		C 0x00000000FFFFFFFF

	mov	PARAM_SIZE, %ecx
	psllq	$32, %mm6		C 0xFFFFFFFF00000000

	psubq	%mm0, %mm6		C 0xFFFFFFFE00000001

	psubq	%mm1, %mm0		C 0xFFFFFFFF - borrow


	movd	(%eax), %mm3		C up
	movd	(%edx), %mm4		C rp

	add	$-1, %ecx
	paddq	%mm6, %mm4		C add 0xFFFFFFFE00000001
	pmuludq	%mm7, %mm3
	jnz	L(gt1)
	psubq	%mm3, %mm4		C prod
	paddq	%mm4, %mm0		C borrow
	movd	%mm0, (%edx)		C result
	jmp	L(rt)

L(gt1):	movd	4(%eax), %mm1		C up
	movd	4(%edx), %mm2		C rp

	add	$-1, %ecx
	jz	L(eev)

	ALIGN(16)
L(top):	paddq	%mm6, %mm2		C add 0xFFFFFFFE00000001
	pmuludq	%mm7, %mm1
	psubq	%mm3, %mm4		C prod
	movd	8(%eax), %mm3		C up
	paddq	%mm4, %mm0		C borrow
	movd	8(%edx), %mm4		C rp
	movd	%mm0, (%edx)		C result
	psrlq	$32, %mm0

	add	$-1, %ecx
	jz	L(eod)

	paddq	%mm6, %mm4		C add 0xFFFFFFFE00000001
	pmuludq	%mm7, %mm3
	psubq	%mm1, %mm2		C prod
	movd	12(%eax), %mm1		C up
	paddq	%mm2, %mm0		C borrow
	movd	12(%edx), %mm2		C rp
	movd	%mm0, 4(%edx)		C result
	psrlq	$32, %mm0

	lea	8(%eax), %eax
	lea	8(%edx), %edx
	add	$-1, %ecx
	jnz	L(top)


L(eev):	paddq	%mm6, %mm2		C add 0xFFFFFFFE00000001
	pmuludq	%mm7, %mm1
	psubq	%mm3, %mm4		C prod
	paddq	%mm4, %mm0		C borrow
	movd	%mm0, (%edx)		C result
	psrlq	$32, %mm0
	psubq	%mm1, %mm2		C prod
	paddq	%mm2, %mm0		C borrow
	movd	%mm0, 4(%edx)		C result
L(rt):	psrlq	$32, %mm0
	movd	%mm0, %eax
	not	%eax
	emms
	ret

L(eod):	paddq	%mm6, %mm4		C add 0xFFFFFFFE00000001
	pmuludq	%mm7, %mm3
	psubq	%mm1, %mm2		C prod
	paddq	%mm2, %mm0		C borrow
	movd	%mm0, 4(%edx)		C result
	psrlq	$32, %mm0
	psubq	%mm3, %mm4		C prod
	paddq	%mm4, %mm0		C borrow
	movd	%mm0, 8(%edx)		C result
	jmp	L(rt)
EPILOGUE()
