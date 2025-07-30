dnl  ARM Neon mpn_com optimised for A15.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

C            cycles/limb
C StrongARM	 ?
C XScale	 ?
C Cortex-A8	 ?
C Cortex-A9	 2.1
C Cortex-A15	 0.65

define(`rp', `r0')
define(`up', `r1')
define(`n',  `r2')

ASM_START()
PROLOGUE(mpn_com)
	cmp		n, #7
	ble		L(bc)

C Perform a few initial operation until rp is 128-bit aligned
	tst		rp, #4
	beq		L(al1)
	vld1.32		{d0[0]}, [up]!
	sub		n, n, #1
	vmvn		d0, d0
	vst1.32		{d0[0]}, [rp]!
L(al1):	tst		rp, #8
	beq		L(al2)
	vld1.32		{d0}, [up]!
	sub		n, n, #2
	vmvn		d0, d0
	vst1.32		{d0}, [rp:64]!
L(al2):	vld1.32		{q2}, [up]!
	subs		n, n, #12
	blt		L(end)

	ALIGN(16)
L(top):	vld1.32		{q0}, [up]!
	vmvn		q2, q2
	subs		n, n, #8
	vst1.32		{q2}, [rp:128]!
	vld1.32		{q2}, [up]!
	vmvn		q0, q0
	vst1.32		{q0}, [rp:128]!
	bge	L(top)

L(end):	vmvn		q2, q2
	vst1.32		{q2}, [rp:128]!

C Handle last 0-7 limbs.  Note that rp is aligned after loop, but not when we
C arrive here via L(bc)
L(bc):	tst		n, #4
	beq		L(tl1)
	vld1.32		{q0}, [up]!
	vmvn		q0, q0
	vst1.32		{q0}, [rp]!
L(tl1):	tst		n, #2
	beq		L(tl2)
	vld1.32		{d0}, [up]!
	vmvn		d0, d0
	vst1.32		{d0}, [rp]!
L(tl2):	tst		n, #1
	beq		L(tl3)
	vld1.32		{d0[0]}, [up]
	vmvn		d0, d0
	vst1.32		{d0[0]}, [rp]
L(tl3):	bx		lr
EPILOGUE()
