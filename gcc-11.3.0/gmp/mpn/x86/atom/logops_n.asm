dnl  Intel Atom mpn_and_n,...,mpn_xnor_n -- bitwise logical operations.

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

C				   cycles/limb
C				op	nop	opn
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 3	 3.5	 3.5
C AMD K6
C AMD K7
C AMD K8
C AMD K10

define(M4_choose_op,
`ifdef(`OPERATION_$1',`
define(`M4_function', `mpn_$1')
define(`M4_want_pre', `$4')
define(`M4_inst',     `$3')
define(`M4_want_post',`$2')
')')
define(M4pre, `ifelse(M4_want_pre, yes,`$1')')
define(M4post,`ifelse(M4_want_post,yes,`$1')')

M4_choose_op( and_n,     , andl,    )
M4_choose_op( andn_n,    , andl, yes)
M4_choose_op( nand_n, yes, andl,    )
M4_choose_op( ior_n,     ,  orl,    )
M4_choose_op( iorn_n,    ,  orl, yes)
M4_choose_op( nior_n, yes,  orl,    )
M4_choose_op( xor_n,     , xorl,    )
M4_choose_op( xnor_n, yes, xorl,    )

ifdef(`M4_function',,
`m4_error(`Unrecognised or undefined OPERATION symbol
')')

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

C void M4_function (mp_ptr dst, mp_srcptr src2, mp_srcptr src1, mp_size_t size);
C

defframe(PARAM_SIZE, 16)
defframe(PARAM_SRC1, 12)
defframe(PARAM_SRC2, 8)
defframe(PARAM_DST,  4)

dnl  re-use parameter space
define(SAVE_RP,`PARAM_SIZE')
define(SAVE_VP,`PARAM_SRC1')
define(SAVE_UP,`PARAM_DST')

define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebx')
define(`cnt', `%eax')
define(`r1',  `%ecx')
define(`r2',  `%edx')

ASM_START()
	TEXT
	ALIGN(16)
deflit(`FRAME',0)

PROLOGUE(M4_function)
	mov	PARAM_SIZE, cnt		C size
	mov	rp, SAVE_RP
	mov	PARAM_DST, rp
	mov	up, SAVE_UP
	mov	PARAM_SRC1, up
	shr	cnt			C size >> 1
	mov	vp, SAVE_VP
	mov	PARAM_SRC2, vp
	mov	(up), r1
	jz	L(end)			C size == 1
	jnc	L(even)			C size % 2 == 0

	ALIGN(16)
L(oop):
M4pre(`	notl_or_xorl_GMP_NUMB_MASK(r1)')
	M4_inst	(vp), r1
	lea	8(up), up
	mov	-4(up), r2
M4post(`	notl_or_xorl_GMP_NUMB_MASK(r1)')
	lea	8(vp), vp
	mov	r1, (rp)
L(entry):
M4pre(`	notl_or_xorl_GMP_NUMB_MASK(r2)')
	M4_inst	-4(vp), r2
	lea	8(rp), rp
M4post(`	notl_or_xorl_GMP_NUMB_MASK(r2)')
	dec	cnt
	mov	(up), r1
	mov	r2, -4(rp)
	jnz	L(oop)

L(end):
M4pre(`	notl_or_xorl_GMP_NUMB_MASK(r1)')
	mov	SAVE_UP, up
	M4_inst	(vp), r1
M4post(`notl_or_xorl_GMP_NUMB_MASK(r1)')
	mov	SAVE_VP, vp
	mov	r1, (rp)
	mov	SAVE_RP, rp
	ret

L(even):
	mov	r1, r2
	lea	4(up), up
	lea	4(vp), vp
	lea	-4(rp), rp
	jmp	L(entry)
EPILOGUE()
ASM_END()
