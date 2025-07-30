dnl  Intel Atom mpn_mul_1.

dnl  Contributed to the GNU project by Torbjorn Granlund and Marco Bodrato.

dnl  Copyright 2011 Free Software Foundation, Inc.

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
C			    cycles/limb
C P5				 -
C P6 model 0-8,10-12		 -
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 7.5
C AMD K6			 -
C AMD K7			 -
C AMD K8
C AMD K10

defframe(PARAM_CARRY,20)
defframe(PARAM_MUL,  16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

define(`rp', `%edx')
define(`up', `%esi')
define(`n',  `%ecx')

ASM_START()
	TEXT
	ALIGN(16)
deflit(`FRAME',0)

PROLOGUE(mpn_mul_1c)
	movd	PARAM_CARRY, %mm6	C carry
	jmp	L(ent)
EPILOGUE()

	ALIGN(8)			C for compact code
PROLOGUE(mpn_mul_1)
	pxor	%mm6, %mm6
L(ent):	push	%esi			FRAME_pushl()
	mov	PARAM_SRC, up
	mov	PARAM_SIZE, %eax	C size
	movd	PARAM_MUL, %mm7
	movd	(up), %mm0
	mov	%eax, n
	and	$3, %eax
	pmuludq	%mm7, %mm0
	mov	PARAM_DST, rp
	jz	L(lo0)
	cmp	$2, %eax
	lea	-16(up,%eax,4),up
	lea	-16(rp,%eax,4),rp
	jc	L(lo1)
	jz	L(lo2)
	jmp	L(lo3)

	ALIGN(16)
L(top):	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	psrlq	$32, %mm6
	lea	16(rp), rp
L(lo0):	paddq	%mm0, %mm6
	movd	4(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, (rp)
	psrlq	$32, %mm6
L(lo3):	paddq	%mm0, %mm6
	movd	8(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, 4(rp)
	psrlq	$32, %mm6
L(lo2):	paddq	%mm0, %mm6
	movd	12(up), %mm0
	pmuludq	%mm7, %mm0
	movd	%mm6, 8(rp)
	psrlq	$32, %mm6
L(lo1):	paddq	%mm0, %mm6
	sub	$4, n
	movd	%mm6, 12(rp)
	lea	16(up), up
	ja	L(top)

	psrlq	$32, %mm6
	movd	%mm6, %eax
	emms
	pop	%esi			FRAME_popl()
	ret
EPILOGUE()
ASM_END()
