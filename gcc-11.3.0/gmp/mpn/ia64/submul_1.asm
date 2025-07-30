dnl  IA-64 mpn_submul_1 -- Multiply a limb vector with a limb and subtract the
dnl  result from a second limb vector.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2000-2004 Free Software Foundation, Inc.

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
C Itanium 2:  2.25 (alignment dependent, sometimes it seems to need 3 c/l)

C TODO
C  * Optimize feed-in and wind-down code, both for speed and code size.
C  * Handle low limb input and results specially, using a common stf8 in the
C    epilogue.
C  * Delay r8, r10 initialization, put cmp-p6 in 1st bundle and br .Ldone in
C    2nd bundle.  This will allow the bbb bundle to be one cycle earlier and
C    save a cycle.

C INPUT PARAMETERS
define(`rp', `r32')
define(`up', `r33')
define(`n',  `r34')
define(`vl', `r35')

ASM_START()
PROLOGUE(mpn_submul_1)
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
	mov		r10 = rp		C M I
	mov		r9 = up			C M I
	sub		vl = r0, vl		C M I	negate vl
}
{.mmi
	ldf8		f8 = [rp], 8		C M
	ldf8		f7 = [up], 8		C M
	add		r19 = -1, n		C M I	n - 1
	;;
}
{.mmi
	cmp.eq		p6, p0 = 0, vl		C M I
	mov		r8 = 0			C M I	zero cylimb
	mov		r2 = ar.lc		C I0
}
{.mmi
	setf.sig	f6 = vl			C M2 M3
	and		r14 = 3, n		C M I
	shr.u		r19 = r19, 2		C I0
	;;
}
{.mmb
	nop		0
	cmp.eq		p10, p0 = 0, r14	C M I
   (p6)	br.spnt		.Ldone			C B	vl == 0
}
{.mmi
	cmp.eq		p11, p0 = 2, r14	C M I
	cmp.eq		p12, p0 = 3, r14	C M I
	mov		ar.lc = r19		C I0
}
{.bbb
  (p10)	br.dptk		.Lb00			C B
  (p11)	br.dptk		.Lb10			C B
  (p12)	br.dptk		.Lb11			C B
	;;
}

.Lb01:	br.cloop.dptk	.grt1

	xma.l		f39 = f7, f6, f8
	xma.hu		f43 = f7, f6, f8
	;;
	getf.sig	r27 = f39			C lo
	getf.sig	r31 = f43			C hi
	ld8		r20 = [r9], 8
	br		.Lcj1

.grt1:	ldf8		f44 = [rp], 8
	ldf8		f32 = [up], 8
	;;
	ldf8		f45 = [rp], 8
	ldf8		f33 = [up], 8
	;;
	ldf8		f46 = [rp], 8
	xma.l		f39 = f7, f6, f8
	ldf8		f34 = [up], 8
	xma.hu		f43 = f7, f6, f8
	;;
	ldf8		f47 = [rp], 8
	xma.l		f36 = f32, f6, f44
	ldf8		f35 = [up], 8
	xma.hu		f40 = f32, f6, f44
	br.cloop.dptk	.grt5
	;;

	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43			C hi
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40			C hi
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41			C hi
	getf.sig	r26 = f38			C lo
	ld8		r23 = [r9], 8
	br		.Lcj5

.grt5:	ldf8		f44 = [rp], 8
	ldf8		f32 = [up], 8
	;;
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f45 = [rp], 8
	getf.sig	r31 = f43			C hi
	ldf8		f33 = [up], 8
	;;
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f46 = [rp], 8
	getf.sig	r28 = f40			C hi
	ldf8		f34 = [up], 8
	;;
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f47 = [rp], 8
	getf.sig	r29 = f41			C hi
	ldf8		f35 = [up], 8
	;;
	getf.sig	r26 = f38			C lo
	xma.l		f36 = f32, f6, f44
	ld8		r23 = [r9], 8
	xma.hu		f40 = f32, f6, f44
	br.cloop.dptk	.Loop
	br		.Lend


.Lb10:	ldf8		f47 = [rp], 8
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt2

	xma.l		f38 = f7, f6, f8
	xma.hu		f42 = f7, f6, f8
	;;
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r26 = f38			C lo
	getf.sig	r30 = f42			C hi
	ld8		r23 = [r9], 8
	;;
	getf.sig	r27 = f39			C lo
	getf.sig	r31 = f43			C hi
	ld8		r20 = [r9], 8
	br		.Lcj2

.grt2:	ldf8		f44 = [rp], 8
	ldf8		f32 = [up], 8
	;;
	ldf8		f45 = [rp], 8
	ldf8		f33 = [up], 8
	xma.l		f38 = f7, f6, f8
	xma.hu		f42 = f7, f6, f8
	;;
	ldf8		f46 = [rp], 8
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f47 = [rp], 8
	ldf8		f35 = [up], 8
	;;
	getf.sig	r26 = f38			C lo
	xma.l		f36 = f32, f6, f44
	ld8		r23 = [r9], 8
	xma.hu		f40 = f32, f6, f44
	br.cloop.dptk	.grt6

	getf.sig	r30 = f42			C hi
	;;
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43			C hi
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40			C hi
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	br		.Lcj6

.grt6:	ldf8		f44 = [rp], 8
	getf.sig	r30 = f42			C hi
	ldf8		f32 = [up], 8
	;;
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f45 = [rp], 8
	getf.sig	r31 = f43			C hi
	ldf8		f33 = [up], 8
	;;
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f46 = [rp], 8
	getf.sig	r28 = f40			C hi
	ldf8		f34 = [up], 8
	;;
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	br		.LL10


.Lb11:	ldf8		f46 = [rp], 8
	ldf8		f34 = [up], 8
	;;
	ldf8		f47 = [rp], 8
	ldf8		f35 = [up], 8
	br.cloop.dptk	.grt3

	xma.l		f37 = f7, f6, f8
	xma.hu		f41 = f7, f6, f8
	;;
	xma.l		f38 = f34, f6, f46
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41			C hi
	ld8		r22 = [r9], 8
	;;
	getf.sig	r26 = f38			C lo
	getf.sig	r30 = f42			C hi
	ld8		r23 = [r9], 8
	;;
	getf.sig	r27 = f39			C lo
	getf.sig	r31 = f43			C hi
	ld8		r20 = [r9], 8
	br		.Lcj3

.grt3:	ldf8		f44 = [rp], 8
	xma.l		f37 = f7, f6, f8
	ldf8		f32 = [up], 8
	xma.hu		f41 = f7, f6, f8
	;;
	ldf8		f45 = [rp], 8
	xma.l		f38 = f34, f6, f46
	ldf8		f33 = [up], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f46 = [rp], 8
	ldf8		f34 = [up], 8
	;;
	getf.sig	r25 = f37			C lo
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f47 = [rp], 8
	getf.sig	r29 = f41			C hi
	ldf8		f35 = [up], 8
	;;
	getf.sig	r26 = f38			C lo
	xma.l		f36 = f32, f6, f44
	ld8		r23 = [r9], 8
	xma.hu		f40 = f32, f6, f44
	br.cloop.dptk	.grt7
	;;

	getf.sig	r30 = f42			C hi
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r31 = f43			C hi
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	br		.Lcj7

.grt7:	ldf8		f44 = [rp], 8
	getf.sig	r30 = f42			C hi
	ldf8		f32 = [up], 8
	;;
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f45 = [rp], 8
	getf.sig	r31 = f43			C hi
	ldf8		f33 = [up], 8
	;;
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	br		.LL11


.Lb00:	ldf8		f45 = [rp], 8
	ldf8		f33 = [up], 8
	;;
	ldf8		f46 = [rp], 8
	ldf8		f34 = [up], 8
	;;
	ldf8		f47 = [rp], 8
	xma.l		f36 = f7, f6, f8
	ldf8		f35 = [up], 8
	xma.hu		f40 = f7, f6, f8
	br.cloop.dptk	.grt4

	xma.l		f37 = f33, f6, f45
	xma.hu		f41 = f33, f6, f45
	;;
	getf.sig	r24 = f36			C lo
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	getf.sig	r28 = f40			C hi
	xma.l		f39 = f35, f6, f47
	getf.sig	r25 = f37			C lo
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	;;
	getf.sig	r29 = f41			C hi
	getf.sig	r26 = f38			C lo
	ld8		r23 = [r9], 8
	;;
	getf.sig	r30 = f42			C hi
	getf.sig	r27 = f39			C lo
	ld8		r20 = [r9], 8
	br		.Lcj4

.grt4:	ldf8		f44 = [rp], 8
	xma.l		f37 = f33, f6, f45
	ldf8		f32 = [up], 8
	xma.hu		f41 = f33, f6, f45
	;;
	ldf8		f45 = [rp], 8
	ldf8		f33 = [up], 8
	xma.l		f38 = f34, f6, f46
	getf.sig	r24 = f36			C lo
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
	;;
	ldf8		f46 = [rp], 8
	getf.sig	r28 = f40			C hi
	ldf8		f34 = [up], 8
	xma.l		f39 = f35, f6, f47
	getf.sig	r25 = f37			C lo
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
	;;
	ldf8		f47 = [rp], 8
	getf.sig	r29 = f41			C hi
	ldf8		f35 = [up], 8
	;;
	getf.sig	r26 = f38			C lo
	xma.l		f36 = f32, f6, f44
	ld8		r23 = [r9], 8
	xma.hu		f40 = f32, f6, f44
	br.cloop.dptk	.grt8
	;;

	getf.sig	r30 = f42			C hi
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	br		.Lcj8

.grt8:	ldf8		f44 = [rp], 8
	getf.sig	r30 = f42			C hi
	ldf8		f32 = [up], 8
	;;
	getf.sig	r27 = f39			C lo
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
	br		.LL00

	ALIGN(32)
.Loop:
{.mmi
	ldf8		f44 = [rp], 8
	cmp.ltu		p6, p0 = r27, r8	C lo cmp
	sub		r14 = r27, r8		C lo sub
}
{.mmi
	getf.sig	r30 = f42			C hi
	ldf8		f32 = [up], 8
	sub		r8 = r20, r31		C hi sub
	;;				C 01
}
{.mmf
	getf.sig	r27 = f39			C lo
	st8		[r10] = r14, 8
	xma.l		f37 = f33, f6, f45
}
{.mfi
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
   (p6)	add		r8 = 1, r8
	;;				C 02
}
{.mmi
.LL00:	ldf8		f45 = [rp], 8
	cmp.ltu		p6, p0 = r24, r8
	sub		r14 = r24, r8
}
{.mmi
	getf.sig	r31 = f43			C hi
	ldf8		f33 = [up], 8
	sub		r8 = r21, r28
	;;				C 03
}
{.mmf
	getf.sig	r24 = f36			C lo
	st8		[r10] = r14, 8
	xma.l		f38 = f34, f6, f46
}
{.mfi
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
   (p6)	add		r8 = 1, r8
	;;				C 04
}
{.mmi
.LL11:	ldf8		f46 = [rp], 8
	cmp.ltu		p6, p0 = r25, r8
	sub		r14 = r25, r8
}
{.mmi
	getf.sig	r28 = f40			C hi
	ldf8		f34 = [up], 8
	sub		r8 = r22, r29
	;;				C 05
}
{.mmf
	getf.sig	r25 = f37			C lo
	st8		[r10] = r14, 8
	xma.l		f39 = f35, f6, f47
}
{.mfi
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
   (p6)	add		r8 = 1, r8
	;;				C 06
}
{.mmi
.LL10:	ldf8		f47 = [rp], 8
	cmp.ltu		p6, p0 = r26, r8
	sub		r14 = r26, r8
}
{.mmi
	getf.sig	r29 = f41			C hi
	ldf8		f35 = [up], 8
	sub		r8 = r23, r30
	;;				C 07
}
{.mmf
	getf.sig	r26 = f38			C lo
	st8		[r10] = r14, 8
	xma.l		f36 = f32, f6, f44
}
{.mfi
	ld8		r23 = [r9], 8
	xma.hu		f40 = f32, f6, f44
   (p6)	add		r8 = 1, r8
}
	br.cloop.dptk	.Loop
	;;

.Lend:
	cmp.ltu		p6, p0 = r27, r8
	sub		r14 = r27, r8
	getf.sig	r30 = f42
	sub		r8 = r20, r31
	;;
	getf.sig	r27 = f39
	st8		[r10] = r14, 8
	xma.l		f37 = f33, f6, f45
	ld8		r20 = [r9], 8
	xma.hu		f41 = f33, f6, f45
   (p6)	add		r8 = 1, r8
	;;
.Lcj8:
	cmp.ltu		p6, p0 = r24, r8
	sub		r14 = r24, r8
	getf.sig	r31 = f43
	sub		r8 = r21, r28
	;;
	getf.sig	r24 = f36
	st8		[r10] = r14, 8
	xma.l		f38 = f34, f6, f46
	ld8		r21 = [r9], 8
	xma.hu		f42 = f34, f6, f46
   (p6)	add		r8 = 1, r8
	;;
.Lcj7:
	cmp.ltu		p6, p0 = r25, r8
	sub		r14 = r25, r8
	getf.sig	r28 = f40
	sub		r8 = r22, r29
	;;
	getf.sig	r25 = f37
	st8		[r10] = r14, 8
	xma.l		f39 = f35, f6, f47
	ld8		r22 = [r9], 8
	xma.hu		f43 = f35, f6, f47
   (p6)	add		r8 = 1, r8
	;;
.Lcj6:
	cmp.ltu		p6, p0 = r26, r8
	sub		r14 = r26, r8
	getf.sig	r29 = f41
	sub		r8 = r23, r30
	;;
	getf.sig	r26 = f38
	st8		[r10] = r14, 8
	ld8		r23 = [r9], 8
   (p6)	add		r8 = 1, r8
	;;
.Lcj5:
	cmp.ltu		p6, p0 = r27, r8
	sub		r14 = r27, r8
	getf.sig	r30 = f42
	sub		r8 = r20, r31
	;;
	getf.sig	r27 = f39
	st8		[r10] = r14, 8
	ld8		r20 = [r9], 8
   (p6)	add		r8 = 1, r8
	;;
.Lcj4:
	cmp.ltu		p6, p0 = r24, r8
	sub		r14 = r24, r8
	getf.sig	r31 = f43
	sub		r8 = r21, r28
	;;
	st8		[r10] = r14, 8
   (p6)	add		r8 = 1, r8
	;;
.Lcj3:
	cmp.ltu		p6, p0 = r25, r8
	sub		r14 = r25, r8
	sub		r8 = r22, r29
	;;
	st8		[r10] = r14, 8
   (p6)	add		r8 = 1, r8
	;;
.Lcj2:
	cmp.ltu		p6, p0 = r26, r8
	sub		r14 = r26, r8
	sub		r8 = r23, r30
	;;
	st8		[r10] = r14, 8
   (p6)	add		r8 = 1, r8
	;;
.Lcj1:
	cmp.ltu		p6, p0 = r27, r8
	sub		r14 = r27, r8
	sub		r8 = r20, r31
	;;
	st8		[r10] = r14, 8
	mov		ar.lc = r2
   (p6)	add		r8 = 1, r8
	br.ret.sptk.many b0
.Ldone:	mov		ar.lc = r2
	br.ret.sptk.many b0
EPILOGUE()
ASM_END()
