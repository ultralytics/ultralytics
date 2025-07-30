dnl  Intel Atom  mpn_bdiv_dbm1.

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
C P6 model 9  (Banias)		 9.75
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)	 8.25
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 8
C AMD K6			 -
C AMD K7			 -
C AMD K8
C AMD K10

C TODO: This code was optimised for atom-32, consider moving it back to atom
C	dir(atom currently grabs this code), and write a 4-way version(7c/l).

defframe(PARAM_CARRY,20)
defframe(PARAM_MUL,  16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

dnl  re-use parameter space
define(SAVE_RP,`PARAM_MUL')
define(SAVE_UP,`PARAM_SIZE')

define(`rp', `%edi')
define(`up', `%esi')
define(`n',  `%ecx')
define(`reg', `%edx')
define(`cy', `%eax')	C contains the return value

ASM_START()
	TEXT
	ALIGN(16)
deflit(`FRAME',0)

PROLOGUE(mpn_bdiv_dbm1c)
	mov	PARAM_SIZE, n		C size
	mov	up, SAVE_UP
	mov	PARAM_SRC, up
	movd	PARAM_MUL, %mm7
	mov	rp, SAVE_RP
	mov	PARAM_DST, rp

	movd	(up), %mm0
	pmuludq	%mm7, %mm0
	shr	n
	mov	PARAM_CARRY, cy
	jz	L(eq1)

	movd	4(up), %mm1
	jc	L(odd)

	lea	4(up), up
	pmuludq	%mm7, %mm1
	movd	%mm0, reg
	psrlq	$32, %mm0
	sub	reg, cy
	movd	%mm0, reg
	movq	%mm1, %mm0
	dec	n
	mov	cy, (rp)
	lea	4(rp), rp
	jz	L(end)

C	ALIGN(16)
L(top):	movd	4(up), %mm1
	sbb	reg, cy
L(odd):	movd	%mm0, reg
	psrlq	$32, %mm0
	pmuludq	%mm7, %mm1
	sub	reg, cy
	lea	8(up), up
	movd	%mm0, reg
	movd	(up), %mm0
	mov	cy, (rp)
	sbb	reg, cy
	movd	%mm1, reg
	psrlq	$32, %mm1
	sub	reg, cy
	movd	%mm1, reg
	pmuludq	%mm7, %mm0
	dec	n
	mov	cy, 4(rp)
	lea	8(rp), rp
	jnz	L(top)

L(end):	sbb	reg, cy

L(eq1):	movd	%mm0, reg
	psrlq	$32, %mm0
	mov	SAVE_UP, up
	sub	reg, cy
	movd	%mm0, reg
	emms
	mov	cy, (rp)
	sbb	reg, cy

	mov	SAVE_RP, rp
	ret
EPILOGUE()
ASM_END()
