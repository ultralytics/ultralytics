dnl  Intel Pentium mpn_lshift -- mpn left shift.

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


C         cycles/limb
C P5,P54:    6.0
C P55:       5.375


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C The main shift-by-N loop should run at 5.375 c/l and that's what P55 does,
C but P5 and P54 run only at 6.0 c/l, which is 4 cycles lost somewhere.

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_lshift)

	pushl	%edi
	pushl	%esi
	pushl	%ebx
	pushl	%ebp
deflit(`FRAME',16)

	movl	PARAM_DST,%edi
	movl	PARAM_SRC,%esi
	movl	PARAM_SIZE,%ebp
	movl	PARAM_SHIFT,%ecx

C We can use faster code for shift-by-1 under certain conditions.
	cmp	$1,%ecx
	jne	L(normal)
	leal	4(%esi),%eax
	cmpl	%edi,%eax
	jnc	L(special)		C jump if s_ptr + 1 >= res_ptr
	leal	(%esi,%ebp,4),%eax
	cmpl	%eax,%edi
	jnc	L(special)		C jump if res_ptr >= s_ptr + size

L(normal):
	leal	-4(%edi,%ebp,4),%edi
	leal	-4(%esi,%ebp,4),%esi

	movl	(%esi),%edx
	subl	$4,%esi
	xorl	%eax,%eax
	shldl(	%cl, %edx, %eax)	C compute carry limb
	pushl	%eax			C push carry limb onto stack

	decl	%ebp
	pushl	%ebp
	shrl	$3,%ebp
	jz	L(end)

	movl	(%edi),%eax		C fetch destination cache line

	ALIGN(4)
L(oop):	movl	-28(%edi),%eax		C fetch destination cache line
	movl	%edx,%ebx

	movl	(%esi),%eax
	movl	-4(%esi),%edx
	shldl(	%cl, %eax, %ebx)
	shldl(	%cl, %edx, %eax)
	movl	%ebx,(%edi)
	movl	%eax,-4(%edi)

	movl	-8(%esi),%ebx
	movl	-12(%esi),%eax
	shldl(	%cl, %ebx, %edx)
	shldl(	%cl, %eax, %ebx)
	movl	%edx,-8(%edi)
	movl	%ebx,-12(%edi)

	movl	-16(%esi),%edx
	movl	-20(%esi),%ebx
	shldl(	%cl, %edx, %eax)
	shldl(	%cl, %ebx, %edx)
	movl	%eax,-16(%edi)
	movl	%edx,-20(%edi)

	movl	-24(%esi),%eax
	movl	-28(%esi),%edx
	shldl(	%cl, %eax, %ebx)
	shldl(	%cl, %edx, %eax)
	movl	%ebx,-24(%edi)
	movl	%eax,-28(%edi)

	subl	$32,%esi
	subl	$32,%edi
	decl	%ebp
	jnz	L(oop)

L(end):	popl	%ebp
	andl	$7,%ebp
	jz	L(end2)
L(oop2):
	movl	(%esi),%eax
	shldl(	%cl,%eax,%edx)
	movl	%edx,(%edi)
	movl	%eax,%edx
	subl	$4,%esi
	subl	$4,%edi
	decl	%ebp
	jnz	L(oop2)

L(end2):
	shll	%cl,%edx		C compute least significant limb
	movl	%edx,(%edi)		C store it

	popl	%eax			C pop carry limb

	popl	%ebp
	popl	%ebx
	popl	%esi
	popl	%edi
	ret


C We loop from least significant end of the arrays, which is only
C permissable if the source and destination don't overlap, since the
C function is documented to work for overlapping source and destination.

L(special):
	movl	(%esi),%edx
	addl	$4,%esi

	decl	%ebp
	pushl	%ebp
	shrl	$3,%ebp

	addl	%edx,%edx
	incl	%ebp
	decl	%ebp
	jz	L(Lend)

	movl	(%edi),%eax		C fetch destination cache line

	ALIGN(4)
L(Loop):
	movl	28(%edi),%eax		C fetch destination cache line
	movl	%edx,%ebx

	movl	(%esi),%eax
	movl	4(%esi),%edx
	adcl	%eax,%eax
	movl	%ebx,(%edi)
	adcl	%edx,%edx
	movl	%eax,4(%edi)

	movl	8(%esi),%ebx
	movl	12(%esi),%eax
	adcl	%ebx,%ebx
	movl	%edx,8(%edi)
	adcl	%eax,%eax
	movl	%ebx,12(%edi)

	movl	16(%esi),%edx
	movl	20(%esi),%ebx
	adcl	%edx,%edx
	movl	%eax,16(%edi)
	adcl	%ebx,%ebx
	movl	%edx,20(%edi)

	movl	24(%esi),%eax
	movl	28(%esi),%edx
	adcl	%eax,%eax
	movl	%ebx,24(%edi)
	adcl	%edx,%edx
	movl	%eax,28(%edi)

	leal	32(%esi),%esi		C use leal not to clobber carry
	leal	32(%edi),%edi
	decl	%ebp
	jnz	L(Loop)

L(Lend):
	popl	%ebp
	sbbl	%eax,%eax		C save carry in %eax
	andl	$7,%ebp
	jz	L(Lend2)
	addl	%eax,%eax		C restore carry from eax
L(Loop2):
	movl	%edx,%ebx
	movl	(%esi),%edx
	adcl	%edx,%edx
	movl	%ebx,(%edi)

	leal	4(%esi),%esi		C use leal not to clobber carry
	leal	4(%edi),%edi
	decl	%ebp
	jnz	L(Loop2)

	jmp	L(L1)
L(Lend2):
	addl	%eax,%eax		C restore carry from eax
L(L1):	movl	%edx,(%edi)		C store last limb

	sbbl	%eax,%eax
	negl	%eax

	popl	%ebp
	popl	%ebx
	popl	%esi
	popl	%edi
	ret

EPILOGUE()
