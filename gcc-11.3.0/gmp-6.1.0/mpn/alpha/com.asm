dnl  Alpha mpn_com -- mpn one's complement.

dnl  Copyright 2003 Free Software Foundation, Inc.

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


C      cycles/limb
C EV4:    4.75
C EV5:    2.0
C EV6:    1.5


C mp_limb_t mpn_com (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C For ev5 the main loop is 7 cycles plus 1 taken branch bubble, for a total
C 2.0 c/l.  In general, a pattern like this unrolled to N limbs per loop
C will be 1.5+2/N c/l.
C
C 2 cycles of loop control are unavoidable, for pointer updates and the
C taken branch bubble, but also since ldq cannot issue two cycles after stq
C (and with a run of stqs that means neither of two cycles at the end of the
C loop.
C
C The fbeq is forced into the second cycle of the loop using unops, since
C the first time through it must wait for the cvtqt result.  Once that
C result is ready (a 1 cycle stall) then both the branch and following loads
C can issue together.
C
C The main loop handles an odd count of limbs, being two limbs loaded before
C each size test, plus one pipelined around from the previous iteration (or
C setup in the entry sequence).
C
C An even number of limbs is handled by an explicit dst[0]=~src[0] in the
C entry sequence, and an increment of the pointers.  For an odd size there's
C no increment and the first store in the loop (r24) is a repeat of dst[0].
C
C Note that the load for r24 after the possible pointer increment is done
C before the explicit store to dst[0], in case src==dst.


ASM_START()

FLOAT64(L(dat), 2.0)

	ALIGN(16)

PROLOGUE(mpn_com,gp)

	C r16	dst
	C r17	src
	C r18	size

	lda	r30, -16(r30)		C temporary stack space
	lda	r7, -3(r18)		C size - 3

	ldq	r20, 0(r17)		C src[0]
	srl	r7, 1, r6		C (size-3)/2

	stq	r6, 8(r30)		C (size-3)/2
	and	r7, 1, r5		C 1 if size even

	LEA(	r8, L(dat))
	s8addq	r5, r17, r17		C skip src[0] if even

	ornot	r31, r20, r20		C ~src[0]
	unop

	ldt	f0, 8(r30)		C (size-3)/2
	ldq	r24, 0(r17)		C src[0 or 1]

	stq	r20, 0(r16)		C dst[0]
	s8addq	r5, r16, r19		C skip dst[0] if even

	ldt	f1, 0(r8)		C data 2.0
	lda	r30, 16(r30)		C restore stack
	unop
	cvtqt	f0, f0			C (size-3)/2 as float

	ornot	r31, r24, r24
	blt	r7, L(done_1)		C if size<=2
	unop
	unop


	C 16-byte alignment here
L(top):
	C r17	src, incrementing
	C r19	dst, incrementing
	C r24	dst[i] result, ready to store
	C f0	(size-3)/2, decrementing
	C f1	2.0

	ldq	r20, 8(r17)		C src[i+1]
	ldq	r21, 16(r17)		C src[i+2]
	unop
	unop

	fbeq	f0, L(done_2)
	unop
	ldq	r22, 24(r17)		C src[i+3]
	ldq	r23, 32(r17)		C src[i+4]

	stq	r24, 0(r19)		C dst[i]
	ornot	r31, r20, r20
	subt	f0, f1, f0		C count -= 2
	unop

	stq	r20, 8(r19)		C dst[i+1]
	ornot	r31, r21, r21
	unop
	unop

	stq	r21, 16(r19)		C dst[i+2]
	ornot	r31, r22, r22

	stq	r22, 24(r19)		C dst[i+3]
	ornot	r31, r23, r24

	lda	r17, 32(r17)		C src += 4
	lda	r19, 32(r19)		C dst += 4
	unop
	fbge	f0, L(top)


L(done_1):
	C r19	&dst[size-1]
	C r24	result for dst[size-1]

	stq	r24, 0(r19)		C dst[size-1]
	ret	r31, (r26), 1


L(done_2):
	C r19	&dst[size-3]
	C r20	src[size-2]
	C r21	src[size-1]
	C r24	result for dst[size-3]

	stq	r24, 0(r19)		C dst[size-3]
	ornot	r31, r20, r20

	stq	r20, 8(r19)		C dst[size-2]
	ornot	r31, r21, r21

	stq	r21, 16(r19)		C dst[size-1]
	ret	r31, (r26), 1

EPILOGUE()
ASM_END()
