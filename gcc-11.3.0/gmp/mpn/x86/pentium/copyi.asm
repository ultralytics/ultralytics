dnl  Intel Pentium mpn_copyi -- copy limb vector, incrementing.

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


C void mpn_copyi (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C Destination prefetching is done to avoid repeated write-throughs on lines
C not already in L1.
C
C At least one of the src or dst pointer needs to be incremented rather than
C using indexing, so that there's somewhere to put the loop control without
C an AGI.  Incrementing one and not two lets us keep loop overhead to 2
C cycles.  Making it the src pointer incremented avoids an AGI on the %ecx
C subtracts in the finishup code.
C
C The block of finishup code is almost as big as the main loop itself, which
C is unfortunate, but it's faster that way than with say rep movsl, by about
C 10 cycles for instance on P55.
C
C There's nothing to be gained from MMX on P55, since it can do only one
C movq load (or store) per cycle, so the throughput would be the same as the
C code here (and even then only if src and dst have the same alignment mod
C 8).

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(8)
PROLOGUE(mpn_copyi)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_DST, %edx

	pushl	%ebx	FRAME_pushl()
	pushl	%esi	FRAME_pushl()

	leal	(%edx,%ecx,4), %edx	C &dst[size-1]
	xorl	$-1, %ecx		C -size-1

	movl	PARAM_SRC, %esi
	addl	$8, %ecx		C -size+7

	jns	L(end)

	movl	-28(%edx,%ecx,4), %eax	C fetch destination cache line, dst[0]
	nop

L(top):
	C eax	scratch
	C ebx	scratch
	C ecx	counter, limbs, negative
	C edx	&dst[size-1]
	C esi	src, incrementing
	C edi
	C ebp

	movl	(%edx,%ecx,4), %eax	C fetch destination cache line
	addl	$8, %ecx

	movl	(%esi), %eax		C read words pairwise
	movl	4(%esi), %ebx
	movl	%eax, -60(%edx,%ecx,4)	C store words pairwise
	movl	%ebx, -56(%edx,%ecx,4)

	movl	8(%esi), %eax
	movl	12(%esi), %ebx
	movl	%eax, -52(%edx,%ecx,4)
	movl	%ebx, -48(%edx,%ecx,4)

	movl	16(%esi), %eax
	movl	20(%esi), %ebx
	movl	%eax, -44(%edx,%ecx,4)
	movl	%ebx, -40(%edx,%ecx,4)

	movl	24(%esi), %eax
	movl	28(%esi), %ebx
	movl	%eax, -36(%edx,%ecx,4)
	movl	%ebx, -32(%edx,%ecx,4)

	leal	32(%esi), %esi
	js	L(top)


L(end):
	C ecx	0 to 7, representing respectively 7 to 0 limbs remaining
	C esi	src end
	C edx	dst, next location to store

	subl	$4, %ecx
	jns	L(no4)

	movl	(%esi), %eax
	movl	4(%esi), %ebx
	movl	%eax, -12(%edx,%ecx,4)
	movl	%ebx, -8(%edx,%ecx,4)

	movl	8(%esi), %eax
	movl	12(%esi), %ebx
	movl	%eax, -4(%edx,%ecx,4)
	movl	%ebx, (%edx,%ecx,4)

	addl	$16, %esi
	addl	$4, %ecx
L(no4):

	subl	$2, %ecx
	jns	L(no2)

	movl	(%esi), %eax
	movl	4(%esi), %ebx
	movl	%eax, -4(%edx,%ecx,4)
	movl	%ebx, (%edx,%ecx,4)

	addl	$8, %esi
	addl	$2, %ecx
L(no2):

	jnz	L(done)

	movl	(%esi), %eax
	movl	%eax, -4(%edx,%ecx,4)	C risk of cache bank clash here

L(done):
	popl	%esi
	popl	%ebx

	ret

EPILOGUE()
