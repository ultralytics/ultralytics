dnl  AMD K7 mpn_add_n/mpn_sub_n -- mpn add or subtract.

dnl  Copyright 1999-2003 Free Software Foundation, Inc.

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


C K7: 1.64 cycles/limb (at 16 limbs/loop).



dnl  K7: UNROLL_COUNT cycles/limb
dnl           8           1.9
dnl          16           1.64
dnl          32           1.7
dnl          64           2.0
dnl  Maximum possible with the current code is 64.

deflit(UNROLL_COUNT, 16)


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
C
C This code runs at 1.64 cycles/limb, which might be the best possible with
C plain integer operations.  Each limb is 2 loads and 1 store, any 2 of
C which can be done each cycle, leading to 1.5 c/l.

dnl  Must have UNROLL_THRESHOLD >= 2, since the unrolled loop can't handle 1.
ifdef(`PIC',`
deflit(UNROLL_THRESHOLD, 8)
',`
deflit(UNROLL_THRESHOLD, 8)
')

defframe(PARAM_CARRY,20)
defframe(PARAM_SIZE, 16)
defframe(PARAM_SRC2, 12)
defframe(PARAM_SRC1, 8)
defframe(PARAM_DST,  4)

defframe(SAVE_EBP, -4)
defframe(SAVE_ESI, -8)
defframe(SAVE_EBX, -12)
defframe(SAVE_EDI, -16)
deflit(STACK_SPACE, 16)

	TEXT
	ALIGN(32)
deflit(`FRAME',0)

PROLOGUE(M4_function_nc)
	movl	PARAM_CARRY, %eax
	jmp	L(start)
EPILOGUE()

PROLOGUE(M4_function_n)

	xorl	%eax, %eax	C carry
L(start):
	movl	PARAM_SIZE, %ecx
	subl	$STACK_SPACE, %esp
deflit(`FRAME',STACK_SPACE)

	movl	%edi, SAVE_EDI
	movl	%ebx, SAVE_EBX
	cmpl	$UNROLL_THRESHOLD, %ecx

	movl	PARAM_SRC2, %edx
	movl	PARAM_SRC1, %ebx
	jae	L(unroll)

	movl	PARAM_DST, %edi
	leal	(%ebx,%ecx,4), %ebx
	leal	(%edx,%ecx,4), %edx

	leal	(%edi,%ecx,4), %edi
	negl	%ecx
	shrl	%eax

	C This loop in in a single 16 byte code block already, so no
	C alignment necessary.
L(simple):
	C eax	scratch
	C ebx	src1
	C ecx	counter
	C edx	src2
	C esi
	C edi	dst
	C ebp

	movl	(%ebx,%ecx,4), %eax
	M4_inst	(%edx,%ecx,4), %eax
	movl	%eax, (%edi,%ecx,4)
	incl	%ecx
	jnz	L(simple)

	movl	$0, %eax
	movl	SAVE_EDI, %edi

	movl	SAVE_EBX, %ebx
	setc	%al
	addl	$STACK_SPACE, %esp

	ret


C -----------------------------------------------------------------------------
	C This is at 0x55, close enough to aligned.
L(unroll):
deflit(`FRAME',STACK_SPACE)
	movl	%ebp, SAVE_EBP
	andl	$-2, %ecx		C size low bit masked out
	andl	$1, PARAM_SIZE		C size low bit kept

	movl	%ecx, %edi
	decl	%ecx
	movl	PARAM_DST, %ebp

	shrl	$UNROLL_LOG2, %ecx
	negl	%edi
	movl	%esi, SAVE_ESI

	andl	$UNROLL_MASK, %edi

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(entry) (%edi,%edi,8), %esi	C 9 bytes per
')
	negl	%edi
	shrl	%eax

	leal	ifelse(UNROLL_BYTES,256,128) (%ebx,%edi,4), %ebx
	leal	ifelse(UNROLL_BYTES,256,128) (%edx,%edi,4), %edx
	leal	ifelse(UNROLL_BYTES,256,128) (%ebp,%edi,4), %edi

	jmp	*%esi


ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	leal	(%edi,%edi,8), %esi
	addl	$L(entry)-L(here), %esi
	addl	(%esp), %esi
	ret_internal
')


C -----------------------------------------------------------------------------
	ALIGN(32)
L(top):
	C eax	zero
	C ebx	src1
	C ecx	counter
	C edx	src2
	C esi	scratch (was computed jump)
	C edi	dst
	C ebp	scratch

	leal	UNROLL_BYTES(%edx), %edx

L(entry):
deflit(CHUNK_COUNT, 2)
forloop(i, 0, UNROLL_COUNT/CHUNK_COUNT-1, `
	deflit(`disp0', eval(i*CHUNK_COUNT*4 ifelse(UNROLL_BYTES,256,-128)))
	deflit(`disp1', eval(disp0 + 4))

Zdisp(	movl,	disp0,(%ebx), %esi)
	movl	disp1(%ebx), %ebp
Zdisp(	M4_inst,disp0,(%edx), %esi)
Zdisp(	movl,	%esi, disp0,(%edi))
	M4_inst	disp1(%edx), %ebp
	movl	%ebp, disp1(%edi)
')

	decl	%ecx
	leal	UNROLL_BYTES(%ebx), %ebx
	leal	UNROLL_BYTES(%edi), %edi
	jns	L(top)


	mov	PARAM_SIZE, %esi
	movl	SAVE_EBP, %ebp
	movl	$0, %eax

	decl	%esi
	js	L(even)

	movl	(%ebx), %ecx
	M4_inst	UNROLL_BYTES(%edx), %ecx
	movl	%ecx, (%edi)
L(even):

	movl	SAVE_EDI, %edi
	movl	SAVE_EBX, %ebx
	setc	%al

	movl	SAVE_ESI, %esi
	addl	$STACK_SPACE, %esp

	ret

EPILOGUE()
