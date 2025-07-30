dnl  Intel Pentium mpn_mul_basecase -- mpn by mpn multiplication.

dnl  Copyright 1996, 1998-2000, 2002 Free Software Foundation, Inc.

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


C P5: 14.2 cycles/crossproduct (approx)


C void mpn_mul_basecase (mp_ptr wp,
C                        mp_srcptr xp, mp_size_t xsize,
C                        mp_srcptr yp, mp_size_t ysize);

defframe(PARAM_YSIZE, 20)
defframe(PARAM_YP,    16)
defframe(PARAM_XSIZE, 12)
defframe(PARAM_XP,    8)
defframe(PARAM_WP,    4)

defframe(VAR_COUNTER, -4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_mul_basecase)

	pushl	%eax			C dummy push for allocating stack slot
	pushl	%esi
	pushl	%ebp
	pushl	%edi
deflit(`FRAME',16)

	movl	PARAM_XP,%esi
	movl	PARAM_WP,%edi
	movl	PARAM_YP,%ebp

	movl	(%esi),%eax		C load xp[0]
	mull	(%ebp)			C multiply by yp[0]
	movl	%eax,(%edi)		C store to wp[0]
	movl	PARAM_XSIZE,%ecx	C xsize
	decl	%ecx			C If xsize = 1, ysize = 1 too
	jz	L(done)

	movl	PARAM_XSIZE,%eax
	pushl	%ebx
FRAME_pushl()
	movl	%edx,%ebx
	leal	(%esi,%eax,4),%esi	C make xp point at end
	leal	(%edi,%eax,4),%edi	C offset wp by xsize
	negl	%ecx			C negate j size/index for inner loop
	xorl	%eax,%eax		C clear carry

	ALIGN(8)
L(oop1):	adcl	$0,%ebx
	movl	(%esi,%ecx,4),%eax	C load next limb at xp[j]
	mull	(%ebp)
	addl	%ebx,%eax
	movl	%eax,(%edi,%ecx,4)
	incl	%ecx
	movl	%edx,%ebx
	jnz	L(oop1)

	adcl	$0,%ebx
	movl	PARAM_YSIZE,%eax
	movl	%ebx,(%edi)		C most significant limb of product
	addl	$4,%edi			C increment wp
	decl	%eax
	jz	L(skip)
	movl	%eax,VAR_COUNTER	C set index i to ysize

L(outer):
	addl	$4,%ebp			C make ebp point to next y limb
	movl	PARAM_XSIZE,%ecx
	negl	%ecx
	xorl	%ebx,%ebx

	C code at 0x61 here, close enough to aligned
L(oop2):
	adcl	$0,%ebx
	movl	(%esi,%ecx,4),%eax
	mull	(%ebp)
	addl	%ebx,%eax
	movl	(%edi,%ecx,4),%ebx
	adcl	$0,%edx
	addl	%eax,%ebx
	movl	%ebx,(%edi,%ecx,4)
	incl	%ecx
	movl	%edx,%ebx
	jnz	L(oop2)

	adcl	$0,%ebx

	movl	%ebx,(%edi)
	addl	$4,%edi
	movl	VAR_COUNTER,%eax
	decl	%eax
	movl	%eax,VAR_COUNTER
	jnz	L(outer)

L(skip):
	popl	%ebx
	popl	%edi
	popl	%ebp
	popl	%esi
	addl	$4,%esp
	ret

L(done):
	movl	%edx,4(%edi)	C store to wp[1]
	popl	%edi
	popl	%ebp
	popl	%esi
	popl	%eax		C dummy pop for deallocating stack slot
	ret

EPILOGUE()

