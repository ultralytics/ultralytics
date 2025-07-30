dnl  Intel Atom mpn_addlshC_n/mpn_rsblshC_n -- rp[] = (vp[] << C) +- up[]

dnl  Contributed to the GNU project by Marco Bodrato.

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

C mp_limb_t mpn_addlshC_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                          mp_size_t size);
C mp_limb_t mpn_addlshC_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                           mp_size_t size, mp_limb_t carry);
C mp_limb_t mpn_rsblshC_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                          mp_size_t size);
C mp_limb_t mpn_rsblshC_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                           mp_size_t size, mp_signed_limb_t carry);

C				cycles/limb
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 6
C AMD K6
C AMD K7
C AMD K8
C AMD K10

defframe(PARAM_CORB,	20)
defframe(PARAM_SIZE,	16)
defframe(PARAM_DBLD,	12)
defframe(PARAM_SRC,	 8)
defframe(PARAM_DST,	 4)

dnl  re-use parameter space
define(VAR_COUNT,`PARAM_SIZE')
define(SAVE_EBP,`PARAM_DBLD')
define(SAVE_VP,`PARAM_SRC')
define(SAVE_UP,`PARAM_DST')

define(M, eval(m4_lshift(1,LSH)))
define(`rp',  `%edi')
define(`up',  `%esi')
define(`vp',  `%ebx')

ASM_START()
	TEXT
	ALIGN(8)

PROLOGUE(M4_function_c)
deflit(`FRAME',0)
	movl	PARAM_CORB, %eax
	movl	%eax, %edx
	shr	$LSH, %edx
	andl	$1, %edx
	M4_opp	%edx, %eax
	jmp	L(start_nc)
EPILOGUE()

PROLOGUE(M4_function)
deflit(`FRAME',0)

	xor	%eax, %eax
	xor	%edx, %edx
L(start_nc):
	push	rp			FRAME_pushl()

	mov	PARAM_SIZE, %ecx	C size
	mov	PARAM_DST, rp
	mov	up, SAVE_UP
	incl	%ecx			C size + 1
	mov	PARAM_SRC, up
	mov	vp, SAVE_VP
	shr	%ecx			C (size+1)\2
	mov	PARAM_DBLD, vp
	mov	%ebp, SAVE_EBP
	mov	%ecx, VAR_COUNT
	jnc	L(entry)		C size odd

	shr	%edx			C size even
	mov	(vp), %ecx
	lea	4(vp), vp
	lea	(%eax,%ecx,M), %edx
	mov	%ecx, %eax
	lea	-4(up), up
	lea	-4(rp), rp
	jmp	L(enteven)

	ALIGN(16)
L(oop):
	lea	(%eax,%ecx,M), %ebp
	shr	$RSH, %ecx
	mov	4(vp), %eax
	shr	%edx
	lea	8(vp), vp
	M4_inst	(up), %ebp
	lea	(%ecx,%eax,M), %edx
	mov	%ebp, (rp)
L(enteven):
	M4_inst	4(up), %edx
	lea	8(up), up
	mov	%edx, 4(rp)
	adc	%edx, %edx
	shr	$RSH, %eax
	lea	8(rp), rp
L(entry):
	mov	(vp), %ecx
	decl	VAR_COUNT
	jnz	L(oop)

	lea	(%eax,%ecx,M), %ebp
	shr	$RSH, %ecx
	shr	%edx
	mov	SAVE_VP, vp
	M4_inst	(up), %ebp
	mov	%ecx, %eax
	mov	SAVE_UP, up
	M4_inst	$0, %eax
	mov	%ebp, (rp)
	mov	SAVE_EBP, %ebp
	pop	rp			FRAME_popl()
	ret
EPILOGUE()

ASM_END()
