dnl  x86 mpn_mul_1 (for 386, 486, and Pentium Pro) -- Multiply a limb vector
dnl  with a limb and store the result in a second limb vector.

dnl  Copyright 1992, 1994, 1997-2002, 2005 Free Software Foundation, Inc.

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
C P5				12.5
C P6 model 0-8,10-12		 5.5
C P6 model 9  (Banias)
C P6 model 13 (Dothan)		 5.25
C P4 model 0  (Willamette)	19.0
C P4 model 1  (?)		19.0
C P4 model 2  (Northwood)	19.0
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6			10.5
C AMD K7			 4.5
C AMD K8


C mp_limb_t mpn_mul_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                      mp_limb_t multiplier);

defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_mul_1)
deflit(`FRAME',0)

	pushl	%edi
	pushl	%esi
	pushl	%ebx
	pushl	%ebp
deflit(`FRAME',16)

	movl	PARAM_DST,%edi
	movl	PARAM_SRC,%esi
	movl	PARAM_SIZE,%ecx

	xorl	%ebx,%ebx
	andl	$3,%ecx
	jz	L(end0)

L(oop0):
	movl	(%esi),%eax
	mull	PARAM_MULTIPLIER
	leal	4(%esi),%esi
	addl	%ebx,%eax
	movl	$0,%ebx
	adcl	%ebx,%edx
	movl	%eax,(%edi)
	movl	%edx,%ebx	C propagate carry into cylimb

	leal	4(%edi),%edi
	decl	%ecx
	jnz	L(oop0)

L(end0):
	movl	PARAM_SIZE,%ecx
	shrl	$2,%ecx
	jz	L(end)


	ALIGN(8)
L(oop):	movl	(%esi),%eax
	mull	PARAM_MULTIPLIER
	addl	%eax,%ebx
	movl	$0,%ebp
	adcl	%edx,%ebp

	movl	4(%esi),%eax
	mull	PARAM_MULTIPLIER
	movl	%ebx,(%edi)
	addl	%eax,%ebp	C new lo + cylimb
	movl	$0,%ebx
	adcl	%edx,%ebx

	movl	8(%esi),%eax
	mull	PARAM_MULTIPLIER
	movl	%ebp,4(%edi)
	addl	%eax,%ebx	C new lo + cylimb
	movl	$0,%ebp
	adcl	%edx,%ebp

	movl	12(%esi),%eax
	mull	PARAM_MULTIPLIER
	movl	%ebx,8(%edi)
	addl	%eax,%ebp	C new lo + cylimb
	movl	$0,%ebx
	adcl	%edx,%ebx

	movl	%ebp,12(%edi)

	leal	16(%esi),%esi
	leal	16(%edi),%edi
	decl	%ecx
	jnz	L(oop)

L(end):	movl	%ebx,%eax

	popl	%ebp
	popl	%ebx
	popl	%esi
	popl	%edi
	ret

EPILOGUE()
