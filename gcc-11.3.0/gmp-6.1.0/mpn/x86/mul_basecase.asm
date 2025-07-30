dnl  x86 mpn_mul_basecase -- Multiply two limb vectors and store the result
dnl  in a third limb vector.

dnl  Copyright 1996-2002 Free Software Foundation, Inc.

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


C     cycles/crossproduct
C P5	  15
C P6	   7.5
C K6	  12.5
C K7	   5.5
C P4	  24


C void mpn_mul_basecase (mp_ptr wp,
C                        mp_srcptr xp, mp_size_t xsize,
C                        mp_srcptr yp, mp_size_t ysize);
C
C This was written in a haste since the Pentium optimized code that was used
C for all x86 machines was slow for the Pentium II.  This code would benefit
C from some cleanup.
C
C To shave off some percentage of the run-time, one should make 4 variants
C of the Louter loop, for the four different outcomes of un mod 4.  That
C would avoid Loop0 altogether.  Code expansion would be > 4-fold for that
C part of the function, but since it is not very large, that would be
C acceptable.
C
C The mul loop (at L(oopM)) might need some tweaking.  It's current speed is
C unknown.

defframe(PARAM_YSIZE,20)
defframe(PARAM_YP,   16)
defframe(PARAM_XSIZE,12)
defframe(PARAM_XP,   8)
defframe(PARAM_WP,   4)

defframe(VAR_MULTIPLIER, -4)
defframe(VAR_COUNTER,    -8)
deflit(VAR_STACK_SPACE,  8)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_mul_basecase)
deflit(`FRAME',0)

	subl	$VAR_STACK_SPACE,%esp
	pushl	%esi
	pushl	%ebp
	pushl	%edi
deflit(`FRAME',eval(VAR_STACK_SPACE+12))

	movl	PARAM_XP,%esi
	movl	PARAM_WP,%edi
	movl	PARAM_YP,%ebp

	movl	(%esi),%eax		C load xp[0]
	mull	(%ebp)			C multiply by yp[0]
	movl	%eax,(%edi)		C store to wp[0]
	movl	PARAM_XSIZE,%ecx	C xsize
	decl	%ecx			C If xsize = 1, ysize = 1 too
	jz	L(done)

	pushl	%ebx
FRAME_pushl()
	movl	%edx,%ebx

	leal	4(%esi),%esi
	leal	4(%edi),%edi

L(oopM):
	movl	(%esi),%eax		C load next limb at xp[j]
	leal	4(%esi),%esi
	mull	(%ebp)
	addl	%ebx,%eax
	movl	%edx,%ebx
	adcl	$0,%ebx
	movl	%eax,(%edi)
	leal	4(%edi),%edi
	decl	%ecx
	jnz	L(oopM)

	movl	%ebx,(%edi)		C most significant limb of product
	addl	$4,%edi			C increment wp
	movl	PARAM_XSIZE,%eax
	shll	$2,%eax
	subl	%eax,%edi
	subl	%eax,%esi

	movl	PARAM_YSIZE,%eax	C ysize
	decl	%eax
	jz	L(skip)
	movl	%eax,VAR_COUNTER	C set index i to ysize

L(outer):
	movl	PARAM_YP,%ebp		C yp
	addl	$4,%ebp			C make ebp point to next v limb
	movl	%ebp,PARAM_YP
	movl	(%ebp),%eax		C copy y limb ...
	movl	%eax,VAR_MULTIPLIER	C ... to stack slot
	movl	PARAM_XSIZE,%ecx

	xorl	%ebx,%ebx
	andl	$3,%ecx
	jz	L(end0)

L(oop0):
	movl	(%esi),%eax
	mull	VAR_MULTIPLIER
	leal	4(%esi),%esi
	addl	%ebx,%eax
	movl	$0,%ebx
	adcl	%ebx,%edx
	addl	%eax,(%edi)
	adcl	%edx,%ebx		C propagate carry into cylimb

	leal	4(%edi),%edi
	decl	%ecx
	jnz	L(oop0)

L(end0):
	movl	PARAM_XSIZE,%ecx
	shrl	$2,%ecx
	jz	L(endX)

	ALIGN(8)
L(oopX):
	movl	(%esi),%eax
	mull	VAR_MULTIPLIER
	addl	%eax,%ebx
	movl	$0,%ebp
	adcl	%edx,%ebp

	movl	4(%esi),%eax
	mull	VAR_MULTIPLIER
	addl	%ebx,(%edi)
	adcl	%eax,%ebp	C new lo + cylimb
	movl	$0,%ebx
	adcl	%edx,%ebx

	movl	8(%esi),%eax
	mull	VAR_MULTIPLIER
	addl	%ebp,4(%edi)
	adcl	%eax,%ebx	C new lo + cylimb
	movl	$0,%ebp
	adcl	%edx,%ebp

	movl	12(%esi),%eax
	mull	VAR_MULTIPLIER
	addl	%ebx,8(%edi)
	adcl	%eax,%ebp	C new lo + cylimb
	movl	$0,%ebx
	adcl	%edx,%ebx

	addl	%ebp,12(%edi)
	adcl	$0,%ebx		C propagate carry into cylimb

	leal	16(%esi),%esi
	leal	16(%edi),%edi
	decl	%ecx
	jnz	L(oopX)

L(endX):
	movl	%ebx,(%edi)
	addl	$4,%edi

	C we incremented wp and xp in the loop above; compensate
	movl	PARAM_XSIZE,%eax
	shll	$2,%eax
	subl	%eax,%edi
	subl	%eax,%esi

	movl	VAR_COUNTER,%eax
	decl	%eax
	movl	%eax,VAR_COUNTER
	jnz	L(outer)

L(skip):
	popl	%ebx
	popl	%edi
	popl	%ebp
	popl	%esi
	addl	$8,%esp
	ret

L(done):
	movl	%edx,4(%edi)	   C store to wp[1]
	popl	%edi
	popl	%ebp
	popl	%esi
	addl	$8,%esp
	ret

EPILOGUE()
