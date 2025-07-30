dnl  Intel Pentium mpn_copyd -- copy limb vector, decrementing.

dnl  Copyright 1996, 2001, 2002, 2006 Free Software Foundation, Inc.

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


C P5: 1.25 cycles/limb


C void mpn_copyd (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C See comments in copyi.asm.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_copyd)
deflit(`FRAME',0)

	movl	PARAM_SRC, %eax
	movl	PARAM_SIZE, %ecx

	pushl	%esi	FRAME_pushl()
	pushl	%edi	FRAME_pushl()

	leal	-4(%eax,%ecx,4), %eax		C &src[size-1]
	movl	PARAM_DST, %edx

	subl	$7, %ecx			C size-7
	jle	L(end)

	movl	28-4(%edx,%ecx,4), %esi		C prefetch cache, dst[size-1]
	nop

L(top):
	C eax	src, decrementing
	C ebx
	C ecx	counter, limbs
	C edx	dst
	C esi	scratch
	C edi	scratch
	C ebp

	movl	28-32(%edx,%ecx,4), %esi	C prefetch dst cache line
	subl	$8, %ecx

	movl	(%eax), %esi			C read words pairwise
	movl	-4(%eax), %edi
	movl	%esi, 56(%edx,%ecx,4)		C store words pairwise
	movl	%edi, 52(%edx,%ecx,4)

	movl	-8(%eax), %esi
	movl	-12(%eax), %edi
	movl	%esi, 48(%edx,%ecx,4)
	movl	%edi, 44(%edx,%ecx,4)

	movl	-16(%eax), %esi
	movl	-20(%eax), %edi
	movl	%esi, 40(%edx,%ecx,4)
	movl	%edi, 36(%edx,%ecx,4)

	movl	-24(%eax), %esi
	movl	-28(%eax), %edi
	movl	%esi, 32(%edx,%ecx,4)
	movl	%edi, 28(%edx,%ecx,4)

	leal	-32(%eax), %eax
	jg	L(top)


L(end):
	C ecx	-7 to 0, representing respectively 0 to 7 limbs remaining
	C eax	src end
	C edx	dst, next location to store

	addl	$4, %ecx
	jle	L(no4)

	movl	(%eax), %esi
	movl	-4(%eax), %edi
	movl	%esi, 8(%edx,%ecx,4)
	movl	%edi, 4(%edx,%ecx,4)

	movl	-8(%eax), %esi
	movl	-12(%eax), %edi
	movl	%esi, (%edx,%ecx,4)
	movl	%edi, -4(%edx,%ecx,4)

	subl	$16, %eax
	subl	$4, %ecx
L(no4):

	addl	$2, %ecx
	jle	L(no2)

	movl	(%eax), %esi
	movl	-4(%eax), %edi
	movl	%esi, (%edx,%ecx,4)
	movl	%edi, -4(%edx,%ecx,4)

	subl	$8, %eax
	subl	$2, %ecx
L(no2):

	jnz	L(done)

	movl	(%eax), %ecx
	movl	%ecx, (%edx)	C risk of cache bank clash here

L(done):
	popl	%edi
	popl	%esi

	ret

EPILOGUE()
