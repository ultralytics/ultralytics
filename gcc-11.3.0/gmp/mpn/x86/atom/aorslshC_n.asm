dnl  Intel Atom mpn_addlshC_n/mpn_sublshC_n -- rp[] = up[] +- (vp[] << C)

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

C mp_limb_t mpn_addlshC_n_ip1 (mp_ptr dst, mp_srcptr src, mp_size_t size);
C mp_limb_t mpn_addlshC_nc_ip1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C				mp_limb_t carry);
C mp_limb_t mpn_sublshC_n_ip1 (mp_ptr dst, mp_srcptr src, mp_size_t size,);
C mp_limb_t mpn_sublshC_nc_ip1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C				mp_signed_limb_t borrow);

defframe(PARAM_CORB,	16)
defframe(PARAM_SIZE,	12)
defframe(PARAM_SRC,	 8)
defframe(PARAM_DST,	 4)

C mp_limb_t mpn_addlshC_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                          mp_size_t size,);
C mp_limb_t mpn_addlshC_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                           mp_size_t size, mp_limb_t carry);
C mp_limb_t mpn_sublshC_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                          mp_size_t size,);
C mp_limb_t mpn_sublshC_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                           mp_size_t size, mp_limb_t borrow);

C if src1 == dst, _ip1 is used

C					cycles/limb
C				dst!=src1,src2	dst==src1
C P5
C P6 model 0-8,10-12
C P6 model 9  (Banias)
C P6 model 13 (Dothan)
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C Intel Atom			 7		 6
C AMD K6
C AMD K7
C AMD K8
C AMD K10

defframe(GPARAM_CORB,	20)
defframe(GPARAM_SIZE,	16)
defframe(GPARAM_SRC2,	12)

dnl  re-use parameter space
define(SAVE_EBP,`PARAM_SIZE')
define(SAVE_EBX,`PARAM_SRC')
define(SAVE_UP,`PARAM_DST')

define(M, eval(m4_lshift(1,LSH)))
define(`rp',  `%edi')
define(`up',  `%esi')

ASM_START()
	TEXT
	ALIGN(8)

PROLOGUE(M4_ip_function_c)
deflit(`FRAME',0)
	movl	PARAM_CORB, %ecx
	movl	%ecx, %edx
	shr	$LSH, %edx
	andl	$1, %edx
	M4_opp	%edx, %ecx
	jmp	L(start_nc)
EPILOGUE()

PROLOGUE(M4_ip_function)
deflit(`FRAME',0)

	xor	%ecx, %ecx
	xor	%edx, %edx
L(start_nc):
	push	rp			FRAME_pushl()
	mov	PARAM_DST, rp
	mov	up, SAVE_UP
	mov	PARAM_SRC, up
	mov	%ebx, SAVE_EBX
	mov	PARAM_SIZE, %ebx	C size
L(inplace):
	incl	%ebx			C size + 1
	shr	%ebx			C (size+1)\2
	mov	%ebp, SAVE_EBP
	jnc	L(entry)		C size odd

	add	%edx, %edx		C size even
	mov	%ecx, %ebp
	mov	(up), %ecx
	lea	-4(rp), rp
	lea	(%ebp,%ecx,M), %eax
	lea	4(up), up
	jmp	L(enteven)

	ALIGN(16)
L(oop):
	lea	(%ecx,%eax,M), %ebp
	shr	$RSH, %eax
	mov	4(up), %ecx
	add	%edx, %edx
	lea	8(up), up
	M4_inst	%ebp, (rp)
	lea	(%eax,%ecx,M), %eax

L(enteven):
	M4_inst	%eax, 4(rp)
	lea	8(rp), rp

	sbb	%edx, %edx
	shr	$RSH, %ecx

L(entry):
	mov	(up), %eax
	decl	%ebx
	jnz	L(oop)

	lea	(%ecx,%eax,M), %ebp
	shr	$RSH, %eax
	shr	%edx
	M4_inst	%ebp, (rp)
	mov	SAVE_UP, up
	adc	$0, %eax
	mov	SAVE_EBP, %ebp
	mov	SAVE_EBX, %ebx
	pop	rp			FRAME_popl()
	ret
EPILOGUE()

PROLOGUE(M4_function_c)
deflit(`FRAME',0)
	movl	GPARAM_CORB, %ecx
	movl	%ecx, %edx
	shr	$LSH, %edx
	andl	$1, %edx
	M4_opp	%edx, %ecx
	jmp	L(generic_nc)
EPILOGUE()

PROLOGUE(M4_function)
deflit(`FRAME',0)

	xor	%ecx, %ecx
	xor	%edx, %edx
L(generic_nc):
	push	rp			FRAME_pushl()
	mov	PARAM_DST, rp
	mov	up, SAVE_UP
	mov	PARAM_SRC, up
	cmp	rp, up
	mov	%ebx, SAVE_EBX
	jne	L(general)
	mov	GPARAM_SIZE, %ebx	C size
	mov	GPARAM_SRC2, up
	jmp	L(inplace)

L(general):
	mov	GPARAM_SIZE, %eax	C size
	mov	%ebx, SAVE_EBX
	incl	%eax			C size + 1
	mov	up, %ebx		C vp
	mov	GPARAM_SRC2, up		C up
	shr	%eax			C (size+1)\2
	mov	%ebp, SAVE_EBP
	mov	%eax, GPARAM_SIZE
	jnc	L(entry2)		C size odd

	add	%edx, %edx		C size even
	mov	%ecx, %ebp
	mov	(up), %ecx
	lea	-4(rp), rp
	lea	-4(%ebx), %ebx
	lea	(%ebp,%ecx,M), %eax
	lea	4(up), up
	jmp	L(enteven2)

	ALIGN(16)
L(oop2):
	lea	(%ecx,%eax,M), %ebp
	shr	$RSH, %eax
	mov	4(up), %ecx
	add	%edx, %edx
	lea	8(up), up
	mov	(%ebx), %edx
	M4_inst	%ebp, %edx
	lea	(%eax,%ecx,M), %eax
	mov	%edx, (rp)
L(enteven2):
	mov	4(%ebx), %edx
	lea	8(%ebx), %ebx
	M4_inst	%eax, %edx
	mov	%edx, 4(rp)
	sbb	%edx, %edx
	shr	$RSH, %ecx
	lea	8(rp), rp
L(entry2):
	mov	(up), %eax
	decl	GPARAM_SIZE
	jnz	L(oop2)

	lea	(%ecx,%eax,M), %ebp
	shr	$RSH, %eax
	shr	%edx
	mov	(%ebx), %edx
	M4_inst	%ebp, %edx
	mov	%edx, (rp)
	mov	SAVE_UP, up
	adc	$0, %eax
	mov	SAVE_EBP, %ebp
	mov	SAVE_EBX, %ebx
	pop	rp			FRAME_popl()
	ret
EPILOGUE()

ASM_END()
