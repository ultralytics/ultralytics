dnl  Intel Pentium mpn_addmul_1 -- mpn by limb multiplication.

dnl  Copyright 1992, 1994, 1996, 1999, 2000, 2002 Free Software Foundation,
dnl  Inc.

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


C P5: 14.0 cycles/limb


ifdef(`OPERATION_addmul_1', `
      define(M4_inst,        addl)
      define(M4_function_1,  mpn_addmul_1)
      define(M4_function_1c, mpn_addmul_1c)

',`ifdef(`OPERATION_submul_1', `
      define(M4_inst,        subl)
      define(M4_function_1,  mpn_submul_1)
      define(M4_function_1c, mpn_submul_1c)

',`m4_error(`Need OPERATION_addmul_1 or OPERATION_submul_1
')')')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_addmul_1c mpn_submul_1 mpn_submul_1c)


C mp_limb_t mpn_addmul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                         mp_limb_t mult);
C mp_limb_t mpn_addmul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                          mp_limb_t mult, mp_limb_t carry);
C
C mp_limb_t mpn_submul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                         mp_limb_t mult);
C mp_limb_t mpn_submul_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                          mp_limb_t mult, mp_limb_t carry);
C

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT

	ALIGN(8)
PROLOGUE(M4_function_1c)
deflit(`FRAME',0)

	movl	PARAM_CARRY, %ecx
	pushl	%esi		FRAME_pushl()

	jmp	L(start_1c)

EPILOGUE()


	ALIGN(8)
PROLOGUE(M4_function_1)
deflit(`FRAME',0)

	xorl	%ecx, %ecx
	pushl	%esi		FRAME_pushl()

L(start_1c):
	movl	PARAM_SRC, %esi
	movl	PARAM_SIZE, %eax

	pushl	%edi		FRAME_pushl()
	pushl	%ebx		FRAME_pushl()

	movl	PARAM_DST, %edi
	leal	-1(%eax), %ebx		C size-1

	leal	(%esi,%eax,4), %esi
	xorl	$-1, %ebx		C -size, and clear carry

	leal	(%edi,%eax,4), %edi

L(top):
	C eax
	C ebx	counter, negative
	C ecx	carry
	C edx
	C esi	src end
	C edi	dst end
	C ebp

	adcl	$0, %ecx
	movl	(%esi,%ebx,4), %eax

	mull	PARAM_MULTIPLIER

	addl	%ecx, %eax
	movl	(%edi,%ebx,4), %ecx

	adcl	$0, %edx
	M4_inst	%eax, %ecx

	movl	%ecx, (%edi,%ebx,4)
	incl	%ebx

	movl	%edx, %ecx
	jnz	L(top)


	adcl	$0, %ecx
	popl	%ebx

	movl	%ecx, %eax
	popl	%edi

	popl	%esi

	ret

EPILOGUE()
