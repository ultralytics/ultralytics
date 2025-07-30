dnl  Intel P6 mpn_copyd -- copy limb vector backwards.

dnl  Copyright 2001, 2002 Free Software Foundation, Inc.

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


C P6: 1.75 cycles/limb, or 0.75 if no overlap


C void mpn_copyd (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C An explicit loop is used because a decrementing rep movsl is a bit slow at
C 2.4 c/l.  That rep movsl also has about a 40 cycle startup time, and the
C code here stands a chance of being faster if the branches predict well.
C
C The slightly strange loop form seems necessary for the claimed speed.
C Maybe load/store ordering affects it.
C
C The source and destination are checked to see if they're actually
C overlapping, since it might be possible to use an incrementing rep movsl
C at 0.75 c/l.  (It doesn't suffer the bad startup time of the decrementing
C version.)
C
C Enhancements:
C
C Top speed for an all-integer copy is probably 1.0 c/l, being one load and
C one store each cycle.  Unrolling the loop below would approach 1.0, but
C it'd be good to know why something like store/load/subl + store/load/jnz
C doesn't already run at 1.0 c/l.  It looks like it should decode in 2
C cycles, but doesn't run that way.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

dnl  re-using parameter space
define(SAVE_ESI,`PARAM_SIZE')
define(SAVE_EDI,`PARAM_SRC')

	TEXT
	ALIGN(16)

PROLOGUE(mpn_copyd)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx

	movl	%esi, SAVE_ESI
	movl	PARAM_SRC, %esi

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	subl	$1, %ecx
	jb	L(zero)

	movl	(%esi,%ecx,4), %eax		C src[size-1]
	jz	L(one)

	movl	-4(%esi,%ecx,4), %edx		C src[size-2]
	subl	$2, %ecx
	jbe	L(done_loop)			C 2 or 3 limbs only


	C The usual overlap is
	C
	C     high                   low
	C     +------------------+
	C     |               dst|
	C     +------------------+
	C           +------------------+
	C           |               src|
	C           +------------------+
	C
	C We can use an incrementing copy in the following circumstances.
	C
	C     src+4*size<=dst, since then the regions are disjoint
	C
	C     src==dst, clearly (though this shouldn't occur normally)
	C
	C     src>dst, since in that case it's a requirement of the
	C              parameters that src>=dst+size*4, and hence the
	C              regions are disjoint
	C

	leal	(%edi,%ecx,4), %edx
	cmpl	%edi, %esi
	jae	L(use_movsl)		C src >= dst

	cmpl	%edi, %edx
	movl	4(%esi,%ecx,4), %edx	C src[size-2] again
	jbe	L(use_movsl)		C src+4*size <= dst


L(top):
	C eax	prev high limb
	C ebx
	C ecx	counter, size-3 down to 0 or -1, inclusive, by 2s
	C edx	prev low limb
	C esi	src
	C edi	dst
	C ebp

	movl	%eax, 8(%edi,%ecx,4)
	movl	(%esi,%ecx,4), %eax

	movl	%edx, 4(%edi,%ecx,4)
	movl	-4(%esi,%ecx,4), %edx

	subl	$2, %ecx
	jnbe	L(top)


L(done_loop):
	movl	%eax, 8(%edi,%ecx,4)
	movl	%edx, 4(%edi,%ecx,4)

	C copy low limb (needed if size was odd, but will already have been
	C done in the loop if size was even)
	movl	(%esi), %eax
L(one):
	movl	%eax, (%edi)
	movl	SAVE_EDI, %edi
	movl	SAVE_ESI, %esi

	ret


L(use_movsl):
	C eax
	C ebx
	C ecx	size-3
	C edx
	C esi	src
	C edi	dst
	C ebp

	addl	$3, %ecx

	cld		C better safe than sorry, see mpn/x86/README

	rep
	movsl

L(zero):
	movl	SAVE_ESI, %esi
	movl	SAVE_EDI, %edi

	ret

EPILOGUE()
