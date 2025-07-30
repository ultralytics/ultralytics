dnl  Intel Pentium-4 mpn_rsh1add_n -- mpn (x+y)/2

dnl  Copyright 2001-2004 Free Software Foundation, Inc.

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


C        cycles/limb (approx)
C      dst!=src1,2  dst==src1  dst==src2
C P4:      4.5         6.5        6.5


C mp_limb_t mpn_rsh1add_n (mp_ptr wp, mp_srcptr xp, mp_srcptr yp,
C                          mp_size_t size);
C
C The slightly strange combination of indexing and pointer incrementing
C that's used seems to work best.  Not sure why, but for instance leal
C incrementing on %esi is a 1 or 2 cycle slowdown.
C
C The dependent chain is paddq combining the carry and next (shifted) part,
C plus psrlq to move the new carry down.  That, and just 4 mmx instructions
C in total, makes 4 c/l the target speed, which is almost achieved for
C separate src/dst but when src==dst the write combining anomalies slow it
C down.

defframe(PARAM_SIZE, 16)
defframe(PARAM_YP,   12)
defframe(PARAM_XP,   8)
defframe(PARAM_WP,   4)

dnl  re-use parameter space
define(SAVE_EBX,`PARAM_XP')
define(SAVE_ESI,`PARAM_YP')

	TEXT
	ALIGN(8)

PROLOGUE(mpn_rsh1add_n)
deflit(`FRAME',0)

	movl	PARAM_XP, %edx
	movl	%ebx, SAVE_EBX

	movl	PARAM_YP, %ebx
	movl	%esi, SAVE_ESI

	movl	PARAM_WP, %esi

	movd	(%edx), %mm0		C xp[0]

	movd	(%ebx), %mm1		C yp[0]
	movl	PARAM_SIZE, %ecx

	movl	(%edx), %eax		C xp[0]

	addl	(%ebx), %eax		C xp[0]+yp[0]

	paddq	%mm1, %mm0		C xp[0]+yp[0]
	leal	(%esi,%ecx,4), %esi	C wp end
	negl	%ecx			C -size

	psrlq	$1, %mm0		C (xp[0]+yp[0])/2
	and	$1, %eax		C return value, rsh1 bit of xp[0]+yp[0]
	addl	$1, %ecx		C -(size-1)
	jz	L(done)


L(top):
	C eax	return value
	C ebx	yp end
	C ecx	counter, limbs, -(size-1) to -1 inclusive
	C edx	xp end
	C esi	wp end
	C mm0	carry (32 bits)

	movd	4(%edx), %mm1	C xp[i+1]
	movd	4(%ebx), %mm2	C yp[i+1]
	leal	4(%edx), %edx
	leal	4(%ebx), %ebx
	paddq	%mm2, %mm1		C xp[i+1]+yp[i+1]
	psllq	$31, %mm1		C low bit at 31, further 32 above

	paddq	%mm1, %mm0		C 31 and carry from prev add
	movd	%mm0, -4(%esi,%ecx,4)	C low ready to store dst[i]

	psrlq	$32, %mm0		C high becomes new carry

	addl	$1, %ecx
	jnz	L(top)


L(done):
	movd	%mm0, -4(%esi)		C dst[size-1]
	movl	SAVE_EBX, %ebx

	movl	SAVE_ESI, %esi
	emms
	ret

EPILOGUE()
