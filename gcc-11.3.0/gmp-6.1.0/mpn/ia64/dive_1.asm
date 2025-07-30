dnl  IA-64 mpn_divexact_1 -- mpn by limb exact division.

dnl  Contributed to the GNU project by Torbjorn Granlund and Kevin Ryde.

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

C            cycles/limb
C Itanium:      16
C Itanium 2:     8

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n',  `r34')
define(`divisor', `r35')

define(`lshift', `r24')
define(`rshift', `r25')

C This code is a bit messy, and not as similar to mode1o.asm as desired.

C The critical path during initialization is for computing the inverse of the
C divisor.  Since odd divisors are probably common, we conditionally execute
C the initial count_trailing_zeros code and the downshift.

C Possible improvement: Merge more of the feed-in code into the inverse
C computation.

ASM_START()
	.text
	.align	32
.Ltab:
data1	0,0x01, 0,0xAB, 0,0xCD, 0,0xB7, 0,0x39, 0,0xA3, 0,0xC5, 0,0xEF
data1	0,0xF1, 0,0x1B, 0,0x3D, 0,0xA7, 0,0x29, 0,0x13, 0,0x35, 0,0xDF
data1	0,0xE1, 0,0x8B, 0,0xAD, 0,0x97, 0,0x19, 0,0x83, 0,0xA5, 0,0xCF
data1	0,0xD1, 0,0xFB, 0,0x1D, 0,0x87, 0,0x09, 0,0xF3, 0,0x15, 0,0xBF
data1	0,0xC1, 0,0x6B, 0,0x8D, 0,0x77, 0,0xF9, 0,0x63, 0,0x85, 0,0xAF
data1	0,0xB1, 0,0xDB, 0,0xFD, 0,0x67, 0,0xE9, 0,0xD3, 0,0xF5, 0,0x9F
data1	0,0xA1, 0,0x4B, 0,0x6D, 0,0x57, 0,0xD9, 0,0x43, 0,0x65, 0,0x8F
data1	0,0x91, 0,0xBB, 0,0xDD, 0,0x47, 0,0xC9, 0,0xB3, 0,0xD5, 0,0x7F
data1	0,0x81, 0,0x2B, 0,0x4D, 0,0x37, 0,0xB9, 0,0x23, 0,0x45, 0,0x6F
data1	0,0x71, 0,0x9B, 0,0xBD, 0,0x27, 0,0xA9, 0,0x93, 0,0xB5, 0,0x5F
data1	0,0x61, 0,0x0B, 0,0x2D, 0,0x17, 0,0x99, 0,0x03, 0,0x25, 0,0x4F
data1	0,0x51, 0,0x7B, 0,0x9D, 0,0x07, 0,0x89, 0,0x73, 0,0x95, 0,0x3F
data1	0,0x41, 0,0xEB, 0,0x0D, 0,0xF7, 0,0x79, 0,0xE3, 0,0x05, 0,0x2F
data1	0,0x31, 0,0x5B, 0,0x7D, 0,0xE7, 0,0x69, 0,0x53, 0,0x75, 0,0x1F
data1	0,0x21, 0,0xCB, 0,0xED, 0,0xD7, 0,0x59, 0,0xC3, 0,0xE5, 0,0x0F
data1	0,0x11, 0,0x3B, 0,0x5D, 0,0xC7, 0,0x49, 0,0x33, 0,0x55, 0,0xFF


PROLOGUE(mpn_divexact_1)
	.prologue
	.save		ar.lc, r2
	.body

 {.mmi;	add		r8 = -1, divisor	C M0
	nop		0			C M1
	tbit.z		p8, p9 = divisor, 0	C I0
}
ifdef(`HAVE_ABI_32',
`	addp4		rp = 0, rp		C M2  rp extend
	addp4		up = 0, up		C M3  up extend
	sxt4		n = n')			C I1  size extend
	;;
.Lhere:
 {.mmi;	ld8		r20 = [up], 8		C M0  up[0]
  (p8)	andcm		r8 = r8, divisor	C M1
	mov		r15 = ip		C I0  .Lhere
	;;
}{.mii
	.pred.rel "mutex", p8, p9
  (p9)	mov		rshift = 0		C M0
  (p8)	popcnt		rshift = r8		C I0 r8 = cnt_lo_zeros(divisor)
	cmp.eq		p6, p10 = 1, n		C I1
	;;
}{.mii;	add		r9 = .Ltab-.Lhere, r15	C M0
  (p8)	shr.u		divisor = divisor, rshift C I0
	nop		0			C I1
	;;
}{.mmi;	add		n = -4, n		C M0  size-1
  (p10)	ld8		r21 = [up], 8		C M1  up[1]
	mov		r14 = 2			C M1  2
}{.mfi;	setf.sig	f6 = divisor		C M2  divisor
	mov		f9 = f0			C M3  carry		FIXME
	zxt1		r3 = divisor		C I1  divisor low byte
	;;
}{.mmi;	add		r3 = r9, r3		C M0  table offset ip and index
	sub		r16 = 0, divisor	C M1  -divisor
	mov		r2 = ar.lc		C I0
}{.mmi;	sub		lshift = 64, rshift	C M2
	setf.sig	f13 = r14		C M3  2 in significand
	mov		r17 = -1		C I1  -1
	;;
}{.mmi;	ld1		r3 = [r3]		C M0  inverse, 8 bits
	nop		0			C M1
	mov		ar.lc = n		C I0  size-1 loop count
}{.mmi;	setf.sig	f12 = r16		C M2  -divisor
	setf.sig	f8 = r17		C M3  -1
	cmp.eq		p7, p0 = -2, n		C I1
	;;
}{.mmi;	setf.sig	f7 = r3			C M2  inverse, 8 bits
	cmp.eq		p8, p0 = -1, n		C M0
	shr.u		r23 = r20, rshift	C I0
	;;
}

	C f6	divisor
	C f7	inverse, being calculated
	C f8	-1, will be -inverse
	C f9	carry
	C f12	-divisor
	C f13	2
	C f14	scratch

	xmpy.l		f14 = f13, f7		C Newton 2*i
	xmpy.l		f7 = f7, f7		C Newton i*i
	;;
	xma.l		f7 = f7, f12, f14	C Newton i*i*-d + 2*i, 16 bits
	;;
	setf.sig	f10 = r23		C speculative, used iff n = 1
	xmpy.l		f14 = f13, f7		C Newton 2*i
	shl		r22 = r21, lshift	C speculative, used iff n > 1
	xmpy.l		f7 = f7, f7		C Newton i*i
	;;
	or		r31 = r22, r23		C speculative, used iff n > 1
	xma.l		f7 = f7, f12, f14	C Newton i*i*-d + 2*i, 32 bits
	shr.u		r23 = r21, rshift	C speculative, used iff n > 1
	;;
	setf.sig	f11 = r31		C speculative, used iff n > 1
	xmpy.l		f14 = f13, f7		C Newton 2*i
	xmpy.l		f7 = f7, f7		C Newton i*i
	;;
	xma.l		f7 = f7, f12, f14	C Newton i*i*-d + 2*i, 64 bits

  (p7)	br.cond.dptk	.Ln2
  (p10)	br.cond.dptk	.grt3
	;;

.Ln1:	xmpy.l		f12 = f10, f7		C q = ulimb * inverse
	br		.Lx1

.Ln2:
	xmpy.l		f8 = f7, f8		C -inverse = inverse * -1
	xmpy.l		f12 = f11, f7		C q = ulimb * inverse
	setf.sig	f11 = r23
	br		.Lx2

.grt3:
	ld8		r21 = [up], 8		C up[2]
	xmpy.l		f8 = f7, f8		C -inverse = inverse * -1
	;;
	shl		r22 = r21, lshift
	;;
	xmpy.l		f12 = f11, f7		C q = ulimb * inverse
	;;
	or		r31 = r22, r23
	shr.u		r23 = r21, rshift
	;;
	setf.sig	f11 = r31
  (p8)	br.cond.dptk	.Lx3			C branch for n = 3
	;;
	ld8		r21 = [up], 8
	br		.Lent

.Ltop:	ld8		r21 = [up], 8
	xma.l		f12 = f9, f8, f10	C q = c * -inverse + si
	nop.b		0
	;;
.Lent:	add		r16 = 160, up
	shl		r22 = r21, lshift
	nop.b		0
	;;
	stf8		[rp] = f12, 8
	xma.hu		f9 = f12, f6, f9	C c = high(q * divisor + c)
	nop.b		0
	nop.m		0
	xmpy.l		f10 = f11, f7		C si = ulimb * inverse
	nop.b		0
	;;
	or		r31 = r22, r23
	shr.u		r23 = r21, rshift
	nop.b		0
	;;
	lfetch		[r16]
	setf.sig	f11 = r31
	br.cloop.sptk.few.clr .Ltop


	xma.l		f12 = f9, f8, f10	C q = c * -inverse + si
	;;
.Lx3:	stf8		[rp] = f12, 8
	xma.hu		f9 = f12, f6, f9	C c = high(q * divisor + c)
	xmpy.l		f10 = f11, f7		C si = ulimb * inverse
	;;
	setf.sig	f11 = r23
	;;
	xma.l		f12 = f9, f8, f10	C q = c * -inverse + si
	;;
.Lx2:	stf8		[rp] = f12, 8
	xma.hu		f9 = f12, f6, f9	C c = high(q * divisor + c)
	xmpy.l		f10 = f11, f7		C si = ulimb * inverse
	;;
	xma.l		f12 = f9, f8, f10	C q = c * -inverse + si
	;;
.Lx1:	stf8		[rp] = f12, 8
	mov		ar.lc = r2		C I0
	br.ret.sptk.many b0
EPILOGUE()
