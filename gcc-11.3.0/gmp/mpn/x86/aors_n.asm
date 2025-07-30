dnl  x86 mpn_add_n/mpn_sub_n -- mpn addition and subtraction.

dnl  Copyright 1992, 1994-1996, 1999-2002 Free Software Foundation, Inc.

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


C     cycles/limb
C P5	3.375
C P6	3.125
C K6	3.5
C K7	2.25
C P4	8.75


ifdef(`OPERATION_add_n',`
	define(M4_inst,        adcl)
	define(M4_function_n,  mpn_add_n)
	define(M4_function_nc, mpn_add_nc)

',`ifdef(`OPERATION_sub_n',`
	define(M4_inst,        sbbl)
	define(M4_function_n,  mpn_sub_n)
	define(M4_function_nc, mpn_sub_nc)

',`m4_error(`Need OPERATION_add_n or OPERATION_sub_n
')')')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)


C mp_limb_t M4_function_n (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C                          mp_size_t size);
C mp_limb_t M4_function_nc (mp_ptr dst, mp_srcptr src1, mp_srcptr src2,
C	                    mp_size_t size, mp_limb_t carry);

defframe(PARAM_CARRY,20)
defframe(PARAM_SIZE, 16)
defframe(PARAM_SRC2, 12)
defframe(PARAM_SRC1, 8)
defframe(PARAM_DST,  4)

	TEXT
	ALIGN(8)

PROLOGUE(M4_function_nc)
deflit(`FRAME',0)

	pushl	%edi		FRAME_pushl()
	pushl	%esi		FRAME_pushl()

	movl	PARAM_DST,%edi
	movl	PARAM_SRC1,%esi
	movl	PARAM_SRC2,%edx
	movl	PARAM_SIZE,%ecx

	movl	%ecx,%eax
	shrl	$3,%ecx			C compute count for unrolled loop
	negl	%eax
	andl	$7,%eax			C get index where to start loop
	jz	L(oopgo)		C necessary special case for 0
	incl	%ecx			C adjust loop count
	shll	$2,%eax			C adjustment for pointers...
	subl	%eax,%edi		C ... since they are offset ...
	subl	%eax,%esi		C ... by a constant when we ...
	subl	%eax,%edx		C ... enter the loop
	shrl	$2,%eax			C restore previous value

ifdef(`PIC',`
	C Calculate start address in loop for PIC.  Due to limitations in
	C old gas, LF(M4_function_n,oop)-L(0a)-3 cannot be put into the leal
	call	L(0a)
L(0a):	leal	(%eax,%eax,8),%eax
	addl	(%esp),%eax
	addl	$L(oop)-L(0a)-3,%eax
	addl	$4,%esp
',`
	C Calculate start address in loop for non-PIC.
	leal	L(oop)-3(%eax,%eax,8),%eax
')

	C These lines initialize carry from the 5th parameter.  Should be
	C possible to simplify.
	pushl	%ebp		FRAME_pushl()
	movl	PARAM_CARRY,%ebp
	shrl	%ebp			C shift bit 0 into carry
	popl	%ebp		FRAME_popl()

	jmp	*%eax			C jump into loop

EPILOGUE()


	ALIGN(16)
PROLOGUE(M4_function_n)
deflit(`FRAME',0)

	pushl	%edi		FRAME_pushl()
	pushl	%esi		FRAME_pushl()

	movl	PARAM_DST,%edi
	movl	PARAM_SRC1,%esi
	movl	PARAM_SRC2,%edx
	movl	PARAM_SIZE,%ecx

	movl	%ecx,%eax
	shrl	$3,%ecx			C compute count for unrolled loop
	negl	%eax
	andl	$7,%eax			C get index where to start loop
	jz	L(oop)			C necessary special case for 0
	incl	%ecx			C adjust loop count
	shll	$2,%eax			C adjustment for pointers...
	subl	%eax,%edi		C ... since they are offset ...
	subl	%eax,%esi		C ... by a constant when we ...
	subl	%eax,%edx		C ... enter the loop
	shrl	$2,%eax			C restore previous value

ifdef(`PIC',`
	C Calculate start address in loop for PIC.  Due to limitations in
	C some assemblers, L(oop)-L(0b)-3 cannot be put into the leal
	call	L(0b)
L(0b):	leal	(%eax,%eax,8),%eax
	addl	(%esp),%eax
	addl	$L(oop)-L(0b)-3,%eax
	addl	$4,%esp
',`
	C Calculate start address in loop for non-PIC.
	leal	L(oop)-3(%eax,%eax,8),%eax
')
	jmp	*%eax			C jump into loop

L(oopgo):
	pushl	%ebp		FRAME_pushl()
	movl	PARAM_CARRY,%ebp
	shrl	%ebp			C shift bit 0 into carry
	popl	%ebp		FRAME_popl()

	ALIGN(16)
L(oop):	movl	(%esi),%eax
	M4_inst	(%edx),%eax
	movl	%eax,(%edi)
	movl	4(%esi),%eax
	M4_inst	4(%edx),%eax
	movl	%eax,4(%edi)
	movl	8(%esi),%eax
	M4_inst	8(%edx),%eax
	movl	%eax,8(%edi)
	movl	12(%esi),%eax
	M4_inst	12(%edx),%eax
	movl	%eax,12(%edi)
	movl	16(%esi),%eax
	M4_inst	16(%edx),%eax
	movl	%eax,16(%edi)
	movl	20(%esi),%eax
	M4_inst	20(%edx),%eax
	movl	%eax,20(%edi)
	movl	24(%esi),%eax
	M4_inst	24(%edx),%eax
	movl	%eax,24(%edi)
	movl	28(%esi),%eax
	M4_inst	28(%edx),%eax
	movl	%eax,28(%edi)
	leal	32(%edi),%edi
	leal	32(%esi),%esi
	leal	32(%edx),%edx
	decl	%ecx
	jnz	L(oop)

	sbbl	%eax,%eax
	negl	%eax

	popl	%esi
	popl	%edi
	ret

EPILOGUE()
