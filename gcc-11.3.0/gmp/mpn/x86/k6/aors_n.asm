dnl  AMD K6 mpn_add/sub_n -- mpn addition or subtraction.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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


C K6: normal 3.25 cycles/limb, in-place 2.75 cycles/limb.


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
C                          mp_size_t size);
C mp_limb_t M4_function_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C	                      mp_size_t size, mp_limb_t carry);
C
C Calculate src1,size M4_description src2,size, and store the result in
C dst,size.  The return value is the carry bit from the top of the result
C (1 or 0).
C
C The _nc version accepts 1 or 0 for an initial carry into the low limb of
C the calculation.  Note values other than 1 or 0 here will lead to garbage
C results.
C
C Instruction decoding limits a normal dst=src1+src2 operation to 3 c/l, and
C an in-place dst+=src to 2.5 c/l.  The unrolled loops have 1 cycle/loop of
C loop control, which with 4 limbs/loop means an extra 0.25 c/l.

define(PARAM_CARRY, `FRAME+20(%esp)')
define(PARAM_SIZE,  `FRAME+16(%esp)')
define(PARAM_SRC2,  `FRAME+12(%esp)')
define(PARAM_SRC1,  `FRAME+8(%esp)')
define(PARAM_DST,   `FRAME+4(%esp)')
deflit(`FRAME',0)

dnl  minimum 5 because the unrolled code can't handle less
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(32)

PROLOGUE(M4_function_nc)
	movl	PARAM_CARRY, %eax
	jmp	L(start)
EPILOGUE()


PROLOGUE(M4_function_n)
	xorl	%eax, %eax
L(start):
	movl	PARAM_SIZE, %ecx
	pushl	%ebx
FRAME_pushl()

	movl	PARAM_SRC1, %ebx
	pushl	%edi
FRAME_pushl()

	movl	PARAM_SRC2, %edx
	cmpl	$UNROLL_THRESHOLD, %ecx

	movl	PARAM_DST, %edi
	jae	L(unroll)


	shrl	%eax		C initial carry flag

	C offset 0x21 here, close enough to aligned
L(simple):
	C eax	scratch
	C ebx	src1
	C ecx	counter
	C edx	src2
	C esi
	C edi	dst
	C ebp
	C
	C The store to (%edi) could be done with a stosl; it'd be smaller
	C code, but there's no speed gain and a cld would have to be added
	C (per mpn/x86/README).

	movl	(%ebx), %eax
	leal	4(%ebx), %ebx

	M4_inst	(%edx), %eax

	movl	%eax, (%edi)
	leal	4(%edi), %edi

	leal	4(%edx), %edx
	loop	L(simple)


	movl	$0, %eax
	popl	%edi

	setc	%al

	popl	%ebx
	ret


C -----------------------------------------------------------------------------
L(unroll):
	C eax	carry
	C ebx	src1
	C ecx	counter
	C edx	src2
	C esi
	C edi	dst
	C ebp

	cmpl	%edi, %ebx
	pushl	%esi

	je	L(inplace)

ifdef(`OPERATION_add_n',`
	cmpl	%edi, %edx

	je	L(inplace_reverse)
')

	movl	%ecx, %esi

	andl	$-4, %ecx
	andl	$3, %esi

	leal	(%ebx,%ecx,4), %ebx
	leal	(%edx,%ecx,4), %edx
	leal	(%edi,%ecx,4), %edi

	negl	%ecx
	shrl	%eax

	ALIGN(32)
L(normal_top):
	C eax	counter, qwords, negative
	C ebx	src1
	C ecx	scratch
	C edx	src2
	C esi
	C edi	dst
	C ebp

	movl	(%ebx,%ecx,4), %eax
	leal	5(%ecx), %ecx
	M4_inst	-20(%edx,%ecx,4), %eax
	movl	%eax, -20(%edi,%ecx,4)

	movl	4-20(%ebx,%ecx,4), %eax
	M4_inst	4-20(%edx,%ecx,4), %eax
	movl	%eax, 4-20(%edi,%ecx,4)

	movl	8-20(%ebx,%ecx,4), %eax
	M4_inst	8-20(%edx,%ecx,4), %eax
	movl	%eax, 8-20(%edi,%ecx,4)

	movl	12-20(%ebx,%ecx,4), %eax
	M4_inst	12-20(%edx,%ecx,4), %eax
	movl	%eax, 12-20(%edi,%ecx,4)

	loop	L(normal_top)


	decl	%esi
	jz	L(normal_finish_one)
	js	L(normal_done)

	C two or three more limbs

	movl	(%ebx), %eax
	M4_inst	(%edx), %eax
	movl	%eax, (%edi)

	movl	4(%ebx), %eax
	M4_inst	4(%edx), %eax
	decl	%esi
	movl	%eax, 4(%edi)

	jz	L(normal_done)
	movl	$2, %ecx

L(normal_finish_one):
	movl	(%ebx,%ecx,4), %eax
	M4_inst	(%edx,%ecx,4), %eax
	movl	%eax, (%edi,%ecx,4)

L(normal_done):
	popl	%esi
	popl	%edi

	movl	$0, %eax
	popl	%ebx

	setc	%al

	ret


C -----------------------------------------------------------------------------

ifdef(`OPERATION_add_n',`
L(inplace_reverse):
	C dst==src2

	movl	%ebx, %edx
')

L(inplace):
	C eax	initial carry
	C ebx
	C ecx	size
	C edx	src
	C esi
	C edi	dst
	C ebp

	leal	-1(%ecx), %esi
	decl	%ecx

	andl	$-4, %ecx
	andl	$3, %esi

	movl	(%edx), %ebx		C src low limb
	leal	(%edx,%ecx,4), %edx

	leal	(%edi,%ecx,4), %edi
	negl	%ecx

	shrl	%eax


	ALIGN(32)
L(inplace_top):
	C eax
	C ebx	next src limb
	C ecx	size
	C edx	src
	C esi
	C edi	dst
	C ebp

	M4_inst	%ebx, (%edi,%ecx,4)

	movl	4(%edx,%ecx,4), %eax
	leal	5(%ecx), %ecx

	M4_inst	%eax, 4-20(%edi,%ecx,4)

	movl	8-20(%edx,%ecx,4), %eax
	movl	12-20(%edx,%ecx,4), %ebx

	M4_inst	%eax, 8-20(%edi,%ecx,4)
	M4_inst	%ebx, 12-20(%edi,%ecx,4)

	movl	16-20(%edx,%ecx,4), %ebx
	loop	L(inplace_top)


	C now %esi is 0 to 3 representing respectively 1 to 4 limbs more

	M4_inst	%ebx, (%edi)

	decl	%esi
	jz	L(inplace_finish_one)
	js	L(inplace_done)

	C two or three more limbs

	movl	4(%edx), %eax
	movl	8(%edx), %ebx
	M4_inst	%eax, 4(%edi)
	M4_inst	%ebx, 8(%edi)

	decl	%esi
	movl	$2, %ecx

	jz	L(normal_done)

L(inplace_finish_one):
	movl	4(%edx,%ecx,4), %eax
	M4_inst	%eax, 4(%edi,%ecx,4)

L(inplace_done):
	popl	%esi
	popl	%edi

	movl	$0, %eax
	popl	%ebx

	setc	%al

	ret

EPILOGUE()
