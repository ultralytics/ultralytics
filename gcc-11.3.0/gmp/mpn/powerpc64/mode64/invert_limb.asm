dnl  PowerPC-64 mpn_invert_limb -- Invert a normalized limb.

dnl  Copyright 2004-2006, 2008, 2010, 2013 Free Software Foundation, Inc.

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

C                  cycles/limb (approximate)
C POWER3/PPC630         80
C POWER4/PPC970         86
C POWER5                86
C POWER6               170
C POWER7                66

ASM_START()
PROLOGUE(mpn_invert_limb,toc)
	LEAL(	r12, approx_tab)
	srdi	r9, r3, 32
	rlwinm	r9, r9, 10, 23, 30	C (d >> 55) & 0x1fe
	srdi	r10, r3, 24		C d >> 24
	lis	r11, 0x1000
	rldicl	r8, r3, 0, 63		C d mod 2
	addi	r10, r10, 1		C d40
	sldi	r11, r11, 32		C 2^60
	srdi	r7, r3, 1		C d/2
	add	r7, r7, r8		C d63 = ceil(d/2)
	neg	r8, r8			C mask = -(d mod 2)
	lhzx	r0, r9, r12
	mullw	r9, r0, r0		C v0*v0
	sldi	r6, r0, 11		C v0 << 11
	addi	r0, r6, -1		C (v0 << 11) - 1
	mulld	r9, r9, r10		C v0*v0*d40
	srdi	r9, r9, 40		C v0*v0*d40 >> 40
	subf	r9, r9, r0		C v1 = (v0 << 11) - (v0*v0*d40 >> 40) - 1
	mulld	r0, r9, r10		C v1*d40
	sldi	r6, r9, 13		C v1 << 13
	subf	r0, r0, r11		C 2^60 - v1*d40
	mulld	r0, r0, r9		C v1 * (2^60 - v1*d40)
	srdi	r0, r0, 47		C v1 * (2^60 - v1*d40) >> 47
	add	r0, r0, r6		C v2 = (v1 << 13) + (v1 * (2^60 - v1*d40) >> 47)
	mulld	r11, r0, r7		C v2 * d63
	srdi	r10, r0, 1		C v2 >> 1
	sldi	r9, r0, 31		C v2 << 31
	and	r8, r10, r8		C (v2 >> 1) & mask
	subf	r8, r11, r8		C ((v2 >> 1) & mask) - v2 * d63
	mulhdu	r0, r8, r0		C p1 = v2 * (((v2 >> 1) & mask) - v2 * d63)
	srdi	r0, r0, 1		C p1 >> 1
	add	r0, r0, r9		C v3 = (v2 << 31) + (p1 >> 1)
	nop
	mulld	r11, r0, r3
	mulhdu	r9, r0, r3
	addc	r10, r11, r3
	adde	r3, r9, r3
	subf	r3, r3, r0
	blr
EPILOGUE()

DEF_OBJECT(approx_tab)
forloop(i,256,512-1,dnl
`	.short	eval(0x7fd00/i)
')dnl
END_OBJECT(approx_tab)
ASM_END()
