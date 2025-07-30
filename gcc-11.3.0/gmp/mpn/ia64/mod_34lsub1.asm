dnl  IA-64 mpn_mod_34lsub1

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2003-2005, 2010 Free Software Foundation, Inc.

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

C           cycles/limb
C Itanium:      ?
C Itanium 2:    1


C INPUT PARAMETERS
define(`up', `r32')
define(`n',  `r33')

C Some useful aliases for registers we use
define(`u0',`r14') define(`u1',`r15') define(`u2',`r16')
define(`a0',`r17') define(`a1',`r18') define(`a2',`r19')
define(`c0',`r20') define(`c1',`r21') define(`c2',`r22')

C This is a fairly simple-minded implementation.  One could approach 0.67 c/l
C with a more sophisticated implementation.  If we're really crazy, we could
C super-unroll, storing carries just in predicate registers, then copy them to
C a general register, and population count them from there.  That'd bring us
C close to 3 insn/limb, for nearly 0.5 c/l.

C Computing n/3 needs 16 cycles, which is a lot of startup overhead.
C We therefore use a plain while-style loop:
C	add		n = -3, n
C	cmp.le		p9, p0 = 3, n
C  (p9)	br.cond		.Loop
C Alternatively, we could table n/3 for, say, n < 256, and predicate the
C 16-cycle code.

C The summing-up code at the end was written quickly, and could surely be
C vastly improved.

ASM_START()
PROLOGUE(mpn_mod_34lsub1)
	.prologue
	.save	ar.lc, r2
	.body
ifdef(`HAVE_ABI_32',`
	addp4		up = 0, up		C			M I
	nop.m		0
	zxt4		n = n			C			I
	;;
')

ifelse(0,1,`
	movl		r14 = 0xAAAAAAAAAAAAAAAB
	;;
	setf.sig	f6 = r14
	setf.sig	f7 = r33
	;;
	xmpy.hu		f6 = f6, f7
	;;
	getf.sig	r8 = f6
	;;
	shr.u		r8 = r8, 1		C Loop count
	;;
	mov.i		ar.lc = r8
')

	ld8	u0 = [up], 8
	cmp.ne	p9, p0 = 1, n
  (p9)	br	L(gt1)
	;;
	shr.u	r8 = u0, 48
	dep.z	r27 = u0, 0, 48
	;;
	add	r8 = r8, r27
	br.ret.sptk.many b0


L(gt1):
 {.mmi;	nop.m	0
	mov	a0 = 0
	add	n = -2, n
}{.mmi;	mov	c0 = 0
	mov	c1 = 0
	mov	c2 = 0
	;;
}{.mmi;	ld8	u1 = [up], 8
	mov	a1 = 0
	cmp.ltu	p6, p0 = r0, r0		C clear p6
}{.mmb;	cmp.gt	p9, p0 = 3, n
	mov	a2 = 0
  (p9)	br.cond.dptk	L(end)
	;;
}
	ALIGN(32)
L(top):
 {.mmi;	ld8	u2 = [up], 8
  (p6)	add	c0 = 1, c0
	cmp.ltu	p7, p0 = a0, u0
}{.mmb;	sub	a0 = a0, u0
	add	n = -3, n
	nop.b	0
	;;
}{.mmi;	ld8	u0 = [up], 8
  (p7)	add	c1 = 1, c1
	cmp.ltu	p8, p0 = a1, u1
}{.mmb;	sub	a1 = a1, u1
	cmp.le	p9, p0 = 3, n
	nop.b	0
	;;
}{.mmi;	ld8	u1 = [up], 8
  (p8)	add	c2 = 1, c2
	cmp.ltu	p6, p0 = a2, u2
}{.mmb;	sub	a2 = a2, u2
	nop.m	0
dnl	br.cloop.dptk	L(top)
  (p9)	br.cond.dptk	L(top)
	;;
}
L(end):
	cmp.eq	p10, p0 = 0, n
	cmp.eq	p11, p0 = 1, n
  (p10)	br	L(0)

L(2):
 {.mmi;	ld8	u2 = [up], 8
  (p6)	add	c0 = 1, c0
	cmp.ltu	p7, p0 = a0, u0
}{.mmb;	sub	a0 = a0, u0
	nop.m	0
  (p11)	br	L(1)
	;;
}	ld8	u0 = [up], 8
  (p7)	add	c1 = 1, c1
	cmp.ltu	p8, p0 = a1, u1
	sub	a1 = a1, u1
	;;
  (p8)	add	c2 = 1, c2
	cmp.ltu	p6, p0 = a2, u2
	sub	a2 = a2, u2
	;;
  (p6)	add	c0 = 1, c0
	cmp.ltu	p7, p0 = a0, u0
	sub	a0 = a0, u0
	;;
  (p7)	add	c1 = 1, c1
	br	L(com)


L(1):
  (p7)	add	c1 = 1, c1
	cmp.ltu	p8, p0 = a1, u1
	sub	a1 = a1, u1
	;;
  (p8)	add	c2 = 1, c2
	cmp.ltu	p6, p0 = a2, u2
	sub	a2 = a2, u2
	;;
  (p6)	add	c0 = 1, c0
	br	L(com)


L(0):
  (p6)	add	c0 = 1, c0
	cmp.ltu	p7, p0 = a0, u0
	sub	a0 = a0, u0
	;;
  (p7)	add	c1 = 1, c1
	cmp.ltu	p8, p0 = a1, u1
	sub	a1 = a1, u1
	;;
  (p8)	add	c2 = 1, c2

L(com):
C |     a2    |     a1    |     a0    |
C |        |        |        |        |
	shr.u	r24 = a0, 48		C 16 bits
	shr.u	r25 = a1, 32		C 32 bits
	shr.u	r26 = a2, 16		C 48 bits
	;;
	shr.u	r10 = c0, 48		C 16 bits, always zero
	shr.u	r11 = c1, 32		C 32 bits
	shr.u	r30 = c2, 16		C 48 bits
	;;
	dep.z	r27 = a0,  0, 48	C 48 bits
	dep.z	r28 = a1, 16, 32	C 48 bits
	dep.z	r29 = a2, 32, 16	C 48 bits
	dep.z	r31 = c0,  0, 48	C 48 bits
	dep.z	r14 = c1, 16, 32	C 48 bits
	dep.z	r15 = c2, 32, 16	C 48 bits
	;;
 {.mmi;	add	r24 = r24, r25
	add	r26 = r26, r27
	add	r28 = r28, r29
}{.mmi;	add	r10 = r10, r11
	add	r30 = r30, r31
	add	r14 = r14, r15
	;;
}
	movl	r8 = 0xffffffffffff0
	add	r24 = r24, r26
	add	r10 = r10, r30
	;;
	add	r24 = r24, r28
	add	r10 = r10, r14
	;;
	sub	r8 = r8, r24
	;;
	add	r8 = r8, r10
	br.ret.sptk.many b0
EPILOGUE()
ASM_END()
