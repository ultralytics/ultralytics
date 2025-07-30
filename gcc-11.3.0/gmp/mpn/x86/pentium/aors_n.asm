dnl  Intel Pentium mpn_add_n/mpn_sub_n -- mpn addition and subtraction.

dnl  Copyright 1992, 1994-1996, 1999, 2000, 2002 Free Software Foundation, Inc.

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


C P5: 2.375 cycles/limb


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
C                           mp_size_t size, mp_limb_t carry);

defframe(PARAM_CARRY,20)
defframe(PARAM_SIZE, 16)
defframe(PARAM_SRC2, 12)
defframe(PARAM_SRC1, 8)
defframe(PARAM_DST,  4)

	TEXT
	ALIGN(8)
PROLOGUE(M4_function_nc)

	pushl	%edi
	pushl	%esi
	pushl	%ebx
	pushl	%ebp
deflit(`FRAME',16)

	movl	PARAM_DST,%edi
	movl	PARAM_SRC1,%esi
	movl	PARAM_SRC2,%ebp
	movl	PARAM_SIZE,%ecx

	movl	(%ebp),%ebx

	decl	%ecx
	movl	%ecx,%edx
	shrl	$3,%ecx
	andl	$7,%edx
	testl	%ecx,%ecx		C zero carry flag
	jz	L(endgo)

	pushl	%edx
FRAME_pushl()
	movl	PARAM_CARRY,%eax
	shrl	%eax			C shift bit 0 into carry
	jmp	L(oop)

L(endgo):
deflit(`FRAME',16)
	movl	PARAM_CARRY,%eax
	shrl	%eax			C shift bit 0 into carry
	jmp	L(end)

EPILOGUE()


	ALIGN(8)
PROLOGUE(M4_function_n)

	pushl	%edi
	pushl	%esi
	pushl	%ebx
	pushl	%ebp
deflit(`FRAME',16)

	movl	PARAM_DST,%edi
	movl	PARAM_SRC1,%esi
	movl	PARAM_SRC2,%ebp
	movl	PARAM_SIZE,%ecx

	movl	(%ebp),%ebx

	decl	%ecx
	movl	%ecx,%edx
	shrl	$3,%ecx
	andl	$7,%edx
	testl	%ecx,%ecx		C zero carry flag
	jz	L(end)
	pushl	%edx
FRAME_pushl()

	ALIGN(8)
L(oop):	movl	28(%edi),%eax		C fetch destination cache line
	leal	32(%edi),%edi

L(1):	movl	(%esi),%eax
	movl	4(%esi),%edx
	M4_inst	%ebx,%eax
	movl	4(%ebp),%ebx
	M4_inst	%ebx,%edx
	movl	8(%ebp),%ebx
	movl	%eax,-32(%edi)
	movl	%edx,-28(%edi)

L(2):	movl	8(%esi),%eax
	movl	12(%esi),%edx
	M4_inst	%ebx,%eax
	movl	12(%ebp),%ebx
	M4_inst	%ebx,%edx
	movl	16(%ebp),%ebx
	movl	%eax,-24(%edi)
	movl	%edx,-20(%edi)

L(3):	movl	16(%esi),%eax
	movl	20(%esi),%edx
	M4_inst	%ebx,%eax
	movl	20(%ebp),%ebx
	M4_inst	%ebx,%edx
	movl	24(%ebp),%ebx
	movl	%eax,-16(%edi)
	movl	%edx,-12(%edi)

L(4):	movl	24(%esi),%eax
	movl	28(%esi),%edx
	M4_inst	%ebx,%eax
	movl	28(%ebp),%ebx
	M4_inst	%ebx,%edx
	movl	32(%ebp),%ebx
	movl	%eax,-8(%edi)
	movl	%edx,-4(%edi)

	leal	32(%esi),%esi
	leal	32(%ebp),%ebp
	decl	%ecx
	jnz	L(oop)

	popl	%edx
FRAME_popl()
L(end):
	decl	%edx			C test %edx w/o clobbering carry
	js	L(end2)
	incl	%edx
L(oop2):
	leal	4(%edi),%edi
	movl	(%esi),%eax
	M4_inst	%ebx,%eax
	movl	4(%ebp),%ebx
	movl	%eax,-4(%edi)
	leal	4(%esi),%esi
	leal	4(%ebp),%ebp
	decl	%edx
	jnz	L(oop2)
L(end2):
	movl	(%esi),%eax
	M4_inst	%ebx,%eax
	movl	%eax,(%edi)

	sbbl	%eax,%eax
	negl	%eax

	popl	%ebp
	popl	%ebx
	popl	%esi
	popl	%edi
	ret

EPILOGUE()
