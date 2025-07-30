dnl  Intel Atom mpn_add_n/mpn_sub_n -- rp[] = up[] +- vp[].

dnl  Copyright 2011 Free Software Foundation, Inc.

dnl  Contributed to the GNU project by Marco Bodrato.

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
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 3
C AMD K6
C AMD K7
C AMD K8
C AMD K10

ifdef(`OPERATION_add_n', `
	define(M4_inst,        adcl)
	define(M4_function_n,  mpn_add_n)
	define(M4_function_nc, mpn_add_nc)
	define(M4_description, add)
',`ifdef(`OPERATION_sub_n', `
	define(M4_inst,        sbbl)
	define(M4_function_n,  mpn_sub_n)
	define(M4_function_nc, mpn_sub_nc)
	define(M4_description, subtract)
',`m4_error(`Need OPERATION_add_n or OPERATION_sub_n
')')')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

C mp_limb_t M4_function_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                         mp_size_t size);
C mp_limb_t M4_function_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C	                   mp_size_t size, mp_limb_t carry);
C
C Calculate src1,size M4_description src2,size, and store the result in
C dst,size.  The return value is the carry bit from the top of the result (1
C or 0).
C
C The _nc version accepts 1 or 0 for an initial carry into the low limb of
C the calculation.  Note values other than 1 or 0 here will lead to garbage
C results.

defframe(PARAM_CARRY,20)
defframe(PARAM_SIZE, 16)
defframe(PARAM_SRC2, 12)
defframe(PARAM_SRC1, 8)
defframe(PARAM_DST,  4)

dnl  re-use parameter space
define(SAVE_RP,`PARAM_SIZE')
define(SAVE_VP,`PARAM_SRC1')
define(SAVE_UP,`PARAM_DST')

define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebx')
define(`cy',  `%ecx')
define(`r1',  `%ecx')
define(`r2',  `%edx')

ASM_START()
	TEXT
	ALIGN(16)
deflit(`FRAME',0)

PROLOGUE(M4_function_n)
	xor	cy, cy			C carry
L(start):
	mov	PARAM_SIZE, %eax	C size
	mov	rp, SAVE_RP
	mov	PARAM_DST, rp
	mov	up, SAVE_UP
	mov	PARAM_SRC1, up
	shr	%eax			C size >> 1
	mov	vp, SAVE_VP
	mov	PARAM_SRC2, vp
	jz	L(one)			C size == 1
	jc	L(three)		C size % 2 == 1

	shr	cy
	mov	(up), r2
	lea	4(up), up
	lea	4(vp), vp
	lea	-4(rp), rp
	jmp	L(entry)
L(one):
	shr	cy
	mov	(up), r1
	jmp	L(end)
L(three):
	shr	cy
	mov	(up), r1

	ALIGN(16)
L(oop):
	M4_inst	(vp), r1
	lea	8(up), up
	mov	-4(up), r2
	lea	8(vp), vp
	mov	r1, (rp)
L(entry):
	M4_inst	-4(vp), r2
	lea	8(rp), rp
	dec	%eax
	mov	(up), r1
	mov	r2, -4(rp)
	jnz	L(oop)

L(end):					C %eax is zero here
	mov	SAVE_UP, up
	M4_inst	(vp), r1
	mov	SAVE_VP, vp
	mov	r1, (rp)
	adc	%eax, %eax
	mov	SAVE_RP, rp
	ret
EPILOGUE()

PROLOGUE(M4_function_nc)
	mov	PARAM_CARRY, cy		C carry
	jmp	L(start)
EPILOGUE()
ASM_END()
