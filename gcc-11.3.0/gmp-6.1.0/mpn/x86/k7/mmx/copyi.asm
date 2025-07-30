dnl  AMD K7 mpn_copyi -- copy limb vector, incrementing.

dnl  Copyright 1999, 2000, 2002, 2003 Free Software Foundation, Inc.

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


C    alignment dst/src, A=0mod8 N=4mod8
C       A/A   A/N   N/A   N/N
C K7    0.75  1.0   1.0   0.75


C void mpn_copyi (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C Copy src,size to dst,size.
C
C This code at 0.75 or 1.0 c/l is always faster than a plain rep movsl at
C 1.33 c/l.
C
C The K7 can do a 64-bit load and 64-bit store in one cycle (optimization
C guile 22007 appendix B), so 0.5 c/l should be possible, however nothing
C under 0.7 c/l is known.  Apparently only two 32-bit stores can be done in
C one cycle, so perhaps some scheduling is needed to ensure it's a
C load+store in each cycle, not store+store.
C
C If both source and destination are unaligned then one limb is processed at
C the start to make them aligned and so get 0.75 c/l, whereas if they'd been
C used unaligned it would be 1.5 c/l.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

dnl  parameter space reused
define(SAVE_EBX,`PARAM_SIZE')

dnl  minimum 5 since the unrolled code can't handle less than 5
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(32)
PROLOGUE(mpn_copyi)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	%ebx, SAVE_EBX

	movl	PARAM_SRC, %eax
	movl	PARAM_DST, %edx

	cmpl	$UNROLL_THRESHOLD, %ecx
	jae	L(unroll)

	orl	%ecx, %ecx
	jz	L(simple_done)

L(simple):
	C eax	src, incrementing
	C ebx	scratch
	C ecx	counter
	C edx	dst, incrementing
	C
	C this loop is 2 cycles/limb

	movl	(%eax), %ebx
	movl	%ebx, (%edx)
	decl	%ecx
	leal	4(%eax), %eax
	leal	4(%edx), %edx
	jnz	L(simple)

L(simple_done):
	movl	SAVE_EBX, %ebx
	ret


L(unroll):
	movl	%eax, %ebx
	leal	-12(%eax,%ecx,4), %eax	C src end - 12
	subl	$3, %ecx		C size-3

	andl	%edx, %ebx
	leal	(%edx,%ecx,4), %edx	C dst end - 12
	negl	%ecx

	testl	$4, %ebx   C testl to pad code closer to 16 bytes for L(top)
	jz	L(aligned)

	C both src and dst unaligned, process one limb to align them
	movl	(%eax,%ecx,4), %ebx
	movl	%ebx, (%edx,%ecx,4)
	incl	%ecx
L(aligned):


	ALIGN(16)
L(top):
	C eax	src end - 12
	C ebx
	C ecx	counter, negative, limbs
	C edx	dst end - 12

	movq	(%eax,%ecx,4), %mm0
	movq	8(%eax,%ecx,4), %mm1
	addl	$4, %ecx
	movq	%mm0, -16(%edx,%ecx,4)
	movq	%mm1, -16+8(%edx,%ecx,4)
	ja	L(top)		C jump no carry and not zero


	C now %ecx is 0 to 3 representing respectively 3 to 0 limbs remaining

	testb	$2, %cl
	jnz	L(finish_not_two)

	movq	(%eax,%ecx,4), %mm0
	movq	%mm0, (%edx,%ecx,4)
L(finish_not_two):

	testb	$1, %cl
	jnz	L(done)

	movl	8(%eax), %ebx
	movl	%ebx, 8(%edx)

L(done):
	movl	SAVE_EBX, %ebx
	emms
	ret

EPILOGUE()
