dnl  IA-64 mpn_bdiv_dbm1.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2008, 2009 Free Software Foundation, Inc.

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

C         cycles/limb
C Itanium:    4
C Itanium 2:  2

C TODO
C  * Optimize feed-in and wind-down code, both for speed and code size.

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n', `r34')
define(`bd', `r35')

ASM_START()
PROLOGUE(mpn_bdiv_dbm1c)
	.prologue
	.save		ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',
`	addp4		rp = 0, rp		C M I
	addp4		up = 0, up		C M I
	zxt4		n = n			C I
	;;
')
{.mmb
	mov		r15 = r36		C M I
	ldf8		f9 = [up], 8		C M
	nop.b		0			C B
}
.Lcommon:
{.mii
	adds		r16 = -1, n		C M I
	mov		r2 = ar.lc		C I0
	and		r14 = 3, n		C M I
	;;
}
{.mii
	setf.sig	f6 = bd			C M2 M3
	shr.u		r31 = r16, 2		C I0
	cmp.eq		p10, p0 = 0, r14	C M I
}
{.mii
	nop.m		0			C M
	cmp.eq		p11, p0 = 2, r14	C M I
	cmp.eq		p12, p0 = 3, r14	C M I
	;;
}
{.mii
	cmp.ne		p6, p7 = r0, r0		C M I
	mov.i		ar.lc = r31		C I0
	cmp.ne		p8, p9 = r0, r0		C M I
}
{.bbb
  (p10)	br.dptk		.Lb00			C B
  (p11)	br.dptk		.Lb10			C B
  (p12)	br.dptk		.Lb11			C B
	;;
}

.Lb01:	br.cloop.dptk	.grt1
	;;
	xma.l		f38 = f9, f6, f0
	xma.hu		f39 = f9, f6, f0
	;;
	getf.sig	r26 = f38
	getf.sig	r27 = f39
	br		.Lcj1

.grt1:	ldf8		f10 = [r33], 8
	;;
	ldf8		f11 = [r33], 8
	;;
	ldf8		f12 = [r33], 8
	;;
	xma.l		f38 = f9, f6, f0
	xma.hu		f39 = f9, f6, f0
	;;
	ldf8		f13 = [r33], 8
	;;
	xma.l		f32 = f10, f6, f0
	xma.hu		f33 = f10, f6, f0
	br.cloop.dptk	.grt5

	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	getf.sig	r27 = f39
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	getf.sig	r21 = f33
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	br		.Lcj5

.grt5:	ldf8		f10 = [r33], 8
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	getf.sig	r27 = f39
	ldf8		f11 = [r33], 8
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	getf.sig	r21 = f33
	ldf8		f12 = [r33], 8
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	br		.LL01

.Lb10:	ldf8		f13 = [r33], 8
	br.cloop.dptk	.grt2
	;;

	xma.l		f36 = f9, f6, f0
	xma.hu		f37 = f9, f6, f0
	;;
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	getf.sig	r24 = f36
	;;
	getf.sig	r25 = f37
	;;
	getf.sig	r26 = f38
	;;
	getf.sig	r27 = f39
	br		.Lcj2

.grt2:	ldf8		f10 = [r33], 8
	;;
	ldf8		f11 = [r33], 8
	;;
	xma.l		f36 = f9, f6, f0
	xma.hu		f37 = f9, f6, f0
	;;
	ldf8		f12 = [r33], 8
	;;
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	ldf8		f13 = [r33], 8
	;;
	getf.sig	r24 = f36
	xma.l		f32 = f10, f6, f0
	xma.hu		f33 = f10, f6, f0
	br.cloop.dptk	.grt6

	getf.sig	r25 = f37
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	getf.sig	r27 = f39
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	br		.Lcj6

.grt6:	getf.sig	r25 = f37
	ldf8		f10 = [r33], 8
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	getf.sig	r27 = f39
	ldf8		f11 = [r33], 8
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	br		.LL10


.Lb11:	ldf8		f12 = [r33], 8
	;;
	ldf8		f13 = [r33], 8
	br.cloop.dptk	.grt3
	;;

	xma.l		f34 = f9, f6, f0
	xma.hu		f35 = f9, f6, f0
	;;
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	getf.sig	r23 = f35
	;;
	getf.sig	r24 = f36
	;;
	getf.sig	r25 = f37
	;;
	getf.sig	r26 = f38
	br		.Lcj3

.grt3:	ldf8		f10 = [r33], 8
	;;
	xma.l		f34 = f9, f6, f0
	xma.hu		f35 = f9, f6, f0
	;;
	ldf8		f11 = [r33], 8
	;;
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	ldf8		f12 = [r33], 8
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	getf.sig	r23 = f35
	ldf8		f13 = [r33], 8
	;;
	getf.sig	r24 = f36
	xma.l		f32 = f10, f6, f0
	xma.hu		f33 = f10, f6, f0
	br.cloop.dptk	.grt7

	getf.sig	r25 = f37
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	br		.Lcj7

.grt7:	getf.sig	r25 = f37
	ldf8		f10 = [r33], 8
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	br		.LL11


.Lb00:	ldf8		f11 = [r33], 8
	;;
	ldf8		f12 = [r33], 8
	;;
	ldf8		f13 = [r33], 8
	br.cloop.dptk	.grt4
	;;

	xma.l		f32 = f9, f6, f0
	xma.hu		f33 = f9, f6, f0
	;;
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	getf.sig	r21 = f33
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	getf.sig	r23 = f35
	;;
	getf.sig	r24 = f36
	br		.Lcj4

.grt4:	xma.l		f32 = f9, f6, f0
	xma.hu		f33 = f9, f6, f0
	;;
	ldf8		f10 = [r33], 8
	;;
	xma.l		f34 = f11, f6, f0
	xma.hu		f35 = f11, f6, f0
	;;
	ldf8		f11 = [r33], 8
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
	xma.hu		f37 = f12, f6, f0
	;;
	getf.sig	r21 = f33
	ldf8		f12 = [r33], 8
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
	xma.hu		f39 = f13, f6, f0
	;;
	getf.sig	r23 = f35
	ldf8		f13 = [r33], 8
	;;
	getf.sig	r24 = f36
	xma.l		f32 = f10, f6, f0
	xma.hu		f33 = f10, f6, f0
	br.cloop.dptk	.LL00
	br		.Lcj8

C *** MAIN LOOP START ***
	ALIGN(32)
.Ltop:
	.pred.rel "mutex",p6,p7
C	.mfi
	getf.sig	r24 = f36
	xma.l		f32 = f10, f6, f0
  (p6)	sub		r15 = r19, r27, 1
C	.mfi
	st8		[r32] = r19, 8
	xma.hu		f33 = f10, f6, f0
  (p7)	sub		r15 = r19, r27
	;;
.LL00:
C	.mfi
	getf.sig	r25 = f37
	nop.f 0
	cmp.ltu		p6, p7 = r15, r20
C	.mib
	ldf8		f10 = [r33], 8
	sub		r16 = r15, r20
	nop.b 0
	;;

C	.mfi
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
  (p6)	sub		r15 = r16, r21, 1
C	.mfi
	st8		[r32] = r16, 8
	xma.hu		f35 = f11, f6, f0
  (p7)	sub		r15 = r16, r21
	;;
.LL11:
C	.mfi
	getf.sig	r27 = f39
	nop.f 0
	cmp.ltu		p6, p7 = r15, r22
C	.mib
	ldf8		f11 = [r33], 8
	sub		r17 = r15, r22
	nop.b 0
	;;

C	.mfi
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
  (p6)	sub		r15 = r17, r23, 1
C	.mfi
	st8		[r32] = r17, 8
	xma.hu		f37 = f12, f6, f0
  (p7)	sub		r15 = r17, r23
	;;
.LL10:
C	.mfi
	getf.sig	r21 = f33
	nop.f 0
	cmp.ltu		p6, p7 = r15, r24
C	.mib
	ldf8		f12 = [r33], 8
	sub		r18 = r15, r24
	nop.b 0
	;;

C	.mfi
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
  (p6)	sub		r15 = r18, r25, 1
C	.mfi
	st8		[r32] = r18, 8
	xma.hu		f39 = f13, f6, f0
  (p7)	sub		r15 = r18, r25
	;;
.LL01:
C	.mfi
	getf.sig	r23 = f35
	nop.f 0
	cmp.ltu		p6, p7 = r15, r26
C	.mib
	ldf8		f13 = [r33], 8
	sub		r19 = r15, r26
	br.cloop.sptk.few .Ltop
C *** MAIN LOOP END ***
	;;

	getf.sig	r24 = f36
	xma.l		f32 = f10, f6, f0
  (p6)	sub		r15 = r19, r27, 1
	st8		[r32] = r19, 8
	xma.hu		f33 = f10, f6, f0
  (p7)	sub		r15 = r19, r27
	;;
.Lcj8:	getf.sig	r25 = f37
	cmp.ltu		p6, p7 = r15, r20
	sub		r16 = r15, r20
	;;
	getf.sig	r26 = f38
	xma.l		f34 = f11, f6, f0
  (p6)	sub		r15 = r16, r21, 1
	st8		[r32] = r16, 8
	xma.hu		f35 = f11, f6, f0
  (p7)	sub		r15 = r16, r21
	;;
.Lcj7:	getf.sig	r27 = f39
	cmp.ltu		p6, p7 = r15, r22
	sub		r17 = r15, r22
	;;
	getf.sig	r20 = f32
	xma.l		f36 = f12, f6, f0
  (p6)	sub		r15 = r17, r23, 1
	st8		[r32] = r17, 8
	xma.hu		f37 = f12, f6, f0
  (p7)	sub		r15 = r17, r23
	;;
.Lcj6:	getf.sig	r21 = f33
	cmp.ltu		p6, p7 = r15, r24
	sub		r18 = r15, r24
	;;
	getf.sig	r22 = f34
	xma.l		f38 = f13, f6, f0
  (p6)	sub		r15 = r18, r25, 1
	st8		[r32] = r18, 8
	xma.hu		f39 = f13, f6, f0
  (p7)	sub		r15 = r18, r25
	;;
.Lcj5:	getf.sig	r23 = f35
	cmp.ltu		p6, p7 = r15, r26
	sub		r19 = r15, r26
	;;
	getf.sig	r24 = f36
  (p6)	sub		r15 = r19, r27, 1
	st8		[r32] = r19, 8
  (p7)	sub		r15 = r19, r27
	;;
.Lcj4:	getf.sig	r25 = f37
	cmp.ltu		p6, p7 = r15, r20
	sub		r16 = r15, r20
	;;
	getf.sig	r26 = f38
  (p6)	sub		r15 = r16, r21, 1
	st8		[r32] = r16, 8
  (p7)	sub		r15 = r16, r21
	;;
.Lcj3:	getf.sig	r27 = f39
	cmp.ltu		p6, p7 = r15, r22
	sub		r17 = r15, r22
	;;
  (p6)	sub		r15 = r17, r23, 1
	st8		[r32] = r17, 8
  (p7)	sub		r15 = r17, r23
	;;
.Lcj2:	cmp.ltu		p6, p7 = r15, r24
	sub		r18 = r15, r24
	;;
  (p6)	sub		r15 = r18, r25, 1
	st8		[r32] = r18, 8
  (p7)	sub		r15 = r18, r25
	;;
.Lcj1:	cmp.ltu		p6, p7 = r15, r26
	sub		r19 = r15, r26
	;;
  (p6)	sub		r8 = r19, r27, 1
	st8		[r32] = r19
  (p7)	sub		r8 = r19, r27
	mov ar.lc = r2
	br.ret.sptk.many b0
EPILOGUE()
ASM_END()
