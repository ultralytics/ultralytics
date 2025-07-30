dnl  IA-64 mpn_mul_1, mpn_mul_1c -- Multiply a limb vector with a limb and
dnl  store the result in a second limb vector.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2000-2004, 2006, 2007 Free Software Foundation, Inc.

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
C Itanium:    4.0
C Itanium 2:  2.0

C TODO
C  * Further optimize feed-in and wind-down code, both for speed and code size.
C  * Handle low limb input and results specially, using a common stf8 in the
C    epilogue.
C  * Use 1 c/l carry propagation scheme in wind-down code.
C  * Use extra pointer register for `up' to speed up feed-in loads.
C  * Work out final differences with addmul_1.asm.

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n', `r34')
define(`vl', `r35')
define(`cy', `r36')	C for mpn_mul_1c

ASM_START()
PROLOGUE(mpn_mul_1)
	.prologue
	.save	ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',
`	addp4		rp = 0, rp		C M I
	addp4		up = 0, up		C M I
	zxt4		n = n			C I
	;;
')
{.mfi
	adds		r15 = -1, n		C M I
	mov		f9 = f0			C F
	mov.i		r2 = ar.lc		C I0
}
{.mmi
	ldf8		f7 = [up], 8		C M
	nop.m		0			C M
	and		r14 = 3, n		C M I
	;;
}
.Lcommon:
{.mii
	setf.sig	f6 = vl			C M2 M3
	shr.u		r31 = r15, 2		C I0
	cmp.eq		p10, p0 = 0, r14	C M I
}
{.mii
	cmp.eq		p11, p0 = 2, r14	C M I
	cmp.eq		p12, p0 = 3, r14	C M I
	nop.i		0			C I
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

.Lb01:	mov		r20 = 0
	br.cloop.dptk	.grt1			C B

	xma.l		f39 = f7, f6, f9	C F
	xma.hu		f43 = f7, f6, f9	C F
	;;
	getf.sig	r8 = f43		C M2
	stf8		[rp] = f39		C M2 M3
	mov.i		ar.lc = r2		C I0
	br.ret.sptk.many b0			C B

.grt1:
	ldf8		f32 = [up], 8
	;;
	ldf8		f33 = [up], 8
	;;
	ldf8		f34 = [up], 8
	xma.l		f39 = f7, f6, f9
	xma.hu		f43 = f7, f6, f9
	;;
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt5

	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	;;
	stf8		[rp] = f39, 8
	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r21 = f43
	getf.sig	r18 = f36
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r22 = f40
	getf.sig	r19 = f37
	xma.l		f39 = f35, f6, f0
	xma.hu		f43 = f35, f6, f0
	;;
	getf.sig	r23 = f41
	getf.sig	r16 = f38
	br		.Lcj5

.grt5:
	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r17 = f39
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r21 = f43
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f0
	;;
	getf.sig	r18 = f36
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r22 = f40
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f0
	;;
	getf.sig	r19 = f37
	xma.hu		f43 = f35, f6, f0
	br		.LL01


.Lb10:	ldf8		f35 = [up], 8
	mov		r23 = 0
	br.cloop.dptk	.grt2

	xma.l		f38 = f7, f6, f9
	xma.hu		f42 = f7, f6, f9
	;;
	stf8		[rp] = f38, 8
	xma.l		f39 = f35, f6, f42
	xma.hu		f43 = f35, f6, f42
	;;
	getf.sig	r8 = f43
	stf8		[rp] = f39
	mov.i		ar.lc = r2
	br.ret.sptk.many b0


.grt2:
	ldf8		f32 = [up], 8
	;;
	ldf8		f33 = [up], 8
	xma.l		f38 = f7, f6, f9
	xma.hu		f42 = f7, f6, f9
	;;
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f0
	xma.hu		f43 = f35, f6, f0
	;;
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt6

	stf8		[rp] = f38, 8
	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r20 = f42
	getf.sig	r17 = f39
	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r21 = f43
	getf.sig	r18 = f36
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r22 = f40
	getf.sig	r19 = f37
	xma.l		f39 = f35, f6, f0
	xma.hu		f43 = f35, f6, f0
	br		.Lcj6

.grt6:
	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r20 = f42
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f0
	;;
	getf.sig	r17 = f39
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r21 = f43
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f0
	;;
	getf.sig	r18 = f36
	xma.hu		f42 = f34, f6, f0
	br		.LL10


.Lb11:	ldf8		f34 = [up], 8
	mov		r22 = 0
	;;
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt3
	;;

	xma.l		f37 = f7, f6, f9
	xma.hu		f41 = f7, f6, f9
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	xma.l		f39 = f35, f6, f0
	xma.hu		f43 = f35, f6, f0
	;;
	getf.sig	r23 = f41
	stf8		[rp] = f37, 8
	getf.sig	r16 = f38
	getf.sig	r20 = f42
	getf.sig	r17 = f39
	getf.sig	r8 = f43
	br		.Lcj3

.grt3:
	ldf8		f32 = [up], 8
	xma.l		f37 = f7, f6, f9
	xma.hu		f41 = f7, f6, f9
	;;
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r19 = f37
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f0
	xma.hu		f43 = f35, f6, f0
	;;
	getf.sig	r23 = f41
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt7

	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
	getf.sig	r20 = f42
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r17 = f39
	xma.l		f37 = f33, f6, f0
	getf.sig	r21 = f43
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r18 = f36
	st8		[rp] = r19, 8
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	br		.Lcj7

.grt7:
	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r20 = f42
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f0
	;;
	getf.sig	r17 = f39
	xma.hu		f41 = f33, f6, f0
	br		.LL11


.Lb00:	ldf8		f33 = [up], 8
	mov		r21 = 0
	;;
	ldf8		f34 = [up], 8
	;;
	ldf8		f35 = [up], 8
	xma.l		f36 = f7, f6, f9
	xma.hu		f40 = f7, f6, f9
	br.cloop.dptk	.grt4

	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r22 = f40
	stf8		[rp] = f36, 8
	xma.l		f39 = f35, f6, f0
	getf.sig	r19 = f37
	xma.hu		f43 = f35, f6, f0
	;;
	getf.sig	r23 = f41
	getf.sig	r16 = f38
	getf.sig	r20 = f42
	getf.sig	r17 = f39
	br		.Lcj4

.grt4:
	ldf8		f32 = [up], 8
	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	;;
	getf.sig	r18 = f36
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f0
	xma.hu		f42 = f34, f6, f0
	;;
	getf.sig	r22 = f40
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f0
	;;
	getf.sig	r19 = f37
	getf.sig	r23 = f41
	xma.hu		f43 = f35, f6, f0
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt8

	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
	getf.sig	r20 = f42
	xma.hu		f40 = f32, f6, f0
	;;
	getf.sig	r17 = f39
	st8		[rp] = r18, 8
	xma.l		f37 = f33, f6, f0
	xma.hu		f41 = f33, f6, f0
	br		.Lcj8

.grt8:
	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
	xma.hu		f40 = f32, f6, f0
	br		.LL00


C *** MAIN LOOP START ***
	ALIGN(32)
.Loop:
	.pred.rel "mutex",p6,p7
	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
   (p6)	cmp.leu		p8, p9 = r24, r17
	st8		[rp] = r24, 8
	xma.hu		f40 = f32, f6, f0
   (p7)	cmp.ltu		p8, p9 = r24, r17
	;;
.LL00:
	.pred.rel "mutex",p8,p9
	getf.sig	r20 = f42
   (p8)	add		r24 = r18, r21, 1
	nop.b		0
	ldf8		f32 = [up], 8
   (p9)	add		r24 = r18, r21
	nop.b		0
	;;
	.pred.rel "mutex",p8,p9
	getf.sig	r17 = f39
	xma.l		f37 = f33, f6, f0
   (p8)	cmp.leu		p6, p7 = r24, r18
	st8		[rp] = r24, 8
	xma.hu		f41 = f33, f6, f0
   (p9)	cmp.ltu		p6, p7 = r24, r18
	;;
.LL11:
	.pred.rel "mutex",p6,p7
	getf.sig	r21 = f43
   (p6)	add		r24 = r19, r22, 1
	nop.b		0
	ldf8		f33 = [up], 8
   (p7)	add		r24 = r19, r22
	nop.b		0
	;;
	.pred.rel "mutex",p6,p7
	getf.sig	r18 = f36
	xma.l		f38 = f34, f6, f0
   (p6)	cmp.leu		p8, p9 = r24, r19
	st8		[rp] = r24, 8
	xma.hu		f42 = f34, f6, f0
   (p7)	cmp.ltu		p8, p9 = r24, r19
	;;
.LL10:
	.pred.rel "mutex",p8,p9
	getf.sig	r22 = f40
   (p8)	add		r24 = r16, r23, 1
	nop.b		0
	ldf8		f34 = [up], 8
   (p9)	add		r24 = r16, r23
	nop.b		0
	;;
	.pred.rel "mutex",p8,p9
	getf.sig	r19 = f37
	xma.l		f39 = f35, f6, f0
   (p8)	cmp.leu		p6, p7 = r24, r16
	st8		[rp] = r24, 8
	xma.hu		f43 = f35, f6, f0
   (p9)	cmp.ltu		p6, p7 = r24, r16
	;;
.LL01:
	.pred.rel "mutex",p6,p7
	getf.sig	r23 = f41
   (p6)	add		r24 = r17, r20, 1
	nop.b		0
	ldf8		f35 = [up], 8
   (p7)	add		r24 = r17, r20
	br.cloop.dptk	.Loop
C *** MAIN LOOP END ***
	;;

.Lcj9:
	.pred.rel "mutex",p6,p7
	getf.sig	r16 = f38
	xma.l		f36 = f32, f6, f0
   (p6)	cmp.leu		p8, p9 = r24, r17
	st8		[rp] = r24, 8
	xma.hu		f40 = f32, f6, f0
   (p7)	cmp.ltu		p8, p9 = r24, r17
	;;
	.pred.rel "mutex",p8,p9
	getf.sig	r20 = f42
   (p8)	add		r24 = r18, r21, 1
   (p9)	add		r24 = r18, r21
	;;
	.pred.rel "mutex",p8,p9
	getf.sig	r17 = f39
	xma.l		f37 = f33, f6, f0
   (p8)	cmp.leu		p6, p7 = r24, r18
	st8		[rp] = r24, 8
	xma.hu		f41 = f33, f6, f0
   (p9)	cmp.ltu		p6, p7 = r24, r18
	;;
.Lcj8:
	.pred.rel "mutex",p6,p7
	getf.sig	r21 = f43
   (p6)	add		r24 = r19, r22, 1
   (p7)	add		r24 = r19, r22
	;;
	.pred.rel "mutex",p6,p7
	getf.sig	r18 = f36
	xma.l		f38 = f34, f6, f0
   (p6)	cmp.leu		p8, p9 = r24, r19
	st8		[rp] = r24, 8
	xma.hu		f42 = f34, f6, f0
   (p7)	cmp.ltu		p8, p9 = r24, r19
	;;
.Lcj7:
	.pred.rel "mutex",p8,p9
	getf.sig	r22 = f40
   (p8)	add		r24 = r16, r23, 1
   (p9)	add		r24 = r16, r23
	;;
	.pred.rel "mutex",p8,p9
	getf.sig	r19 = f37
	xma.l		f39 = f35, f6, f0
   (p8)	cmp.leu		p6, p7 = r24, r16
	st8		[rp] = r24, 8
	xma.hu		f43 = f35, f6, f0
   (p9)	cmp.ltu		p6, p7 = r24, r16
	;;
.Lcj6:
	.pred.rel "mutex",p6,p7
	getf.sig	r23 = f41
   (p6)	add		r24 = r17, r20, 1
   (p7)	add		r24 = r17, r20
	;;
	.pred.rel "mutex",p6,p7
   (p6)	cmp.leu		p8, p9 = r24, r17
   (p7)	cmp.ltu		p8, p9 = r24, r17
	getf.sig	r16 = f38
	st8		[rp] = r24, 8
	;;
.Lcj5:
	.pred.rel "mutex",p8,p9
	getf.sig	r20 = f42
   (p8)	add		r24 = r18, r21, 1
   (p9)	add		r24 = r18, r21
	;;
	.pred.rel "mutex",p8,p9
   (p8)	cmp.leu		p6, p7 = r24, r18
   (p9)	cmp.ltu		p6, p7 = r24, r18
	getf.sig	r17 = f39
	st8		[rp] = r24, 8
	;;
.Lcj4:
	.pred.rel "mutex",p6,p7
	getf.sig	r8 = f43
   (p6)	add		r24 = r19, r22, 1
   (p7)	add		r24 = r19, r22
	;;
	.pred.rel "mutex",p6,p7
	st8		[rp] = r24, 8
   (p6)	cmp.leu		p8, p9 = r24, r19
   (p7)	cmp.ltu		p8, p9 = r24, r19
	;;
.Lcj3:
	.pred.rel "mutex",p8,p9
   (p8)	add		r24 = r16, r23, 1
   (p9)	add		r24 = r16, r23
	;;
	.pred.rel "mutex",p8,p9
	st8		[rp] = r24, 8
   (p8)	cmp.leu		p6, p7 = r24, r16
   (p9)	cmp.ltu		p6, p7 = r24, r16
	;;
.Lcj2:
	.pred.rel "mutex",p6,p7
   (p6)	add		r24 = r17, r20, 1
   (p7)	add		r24 = r17, r20
	;;
	.pred.rel "mutex",p6,p7
	st8		[rp] = r24, 8
   (p6)	cmp.leu		p8, p9 = r24, r17
   (p7)	cmp.ltu		p8, p9 = r24, r17
	;;
   (p8)	add		r8 = 1, r8
	mov.i		ar.lc = r2
	br.ret.sptk.many b0
EPILOGUE()

PROLOGUE(mpn_mul_1c)
	.prologue
	.save	ar.lc, r2
	.body

ifdef(`HAVE_ABI_32',
`	addp4		rp = 0, rp		C M I
	addp4		up = 0, up		C M I
	zxt4		n = n			C I
	;;
')
{.mmi
	adds		r15 = -1, n		C M I
	setf.sig	f9 = cy			C M2 M3
	mov.i		r2 = ar.lc		C I0
}
{.mmb
	ldf8		f7 = [up], 8		C M
	and		r14 = 3, n		C M I
	br.sptk		.Lcommon
	;;
}
EPILOGUE()
ASM_END()
