; mc88100 mpn_sub_n -- Subtract two limb vectors of the same length > 0 and
; store difference in a third limb vector.

; Copyright 1992, 1994, 1996, 2000 Free Software Foundation, Inc.

;  This file is part of the GNU MP Library.
;
;  The GNU MP Library is free software; you can redistribute it and/or modify
;  it under the terms of either:
;
;    * the GNU Lesser General Public License as published by the Free
;      Software Foundation; either version 3 of the License, or (at your
;      option) any later version.
;
;  or
;
;    * the GNU General Public License as published by the Free Software
;      Foundation; either version 2 of the License, or (at your option) any
;      later version.
;
;  or both in parallel, as here.
;
;  The GNU MP Library is distributed in the hope that it will be useful, but
;  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
;  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
;  for more details.
;
;  You should have received copies of the GNU General Public License and the
;  GNU Lesser General Public License along with the GNU MP Library.  If not,
;  see https://www.gnu.org/licenses/.


; INPUT PARAMETERS
; res_ptr	r2
; s1_ptr	r3
; s2_ptr	r4
; size		r5

; This code has been optimized to run one instruction per clock, avoiding
; load stalls and writeback contention.  As a result, the instruction
; order is not always natural.

; The speed is about 4.6 clocks/limb + 18 clocks/limb-vector on an 88100,
; but on the 88110, it seems to run much slower, 6.6 clocks/limb.

	text
	align	 16
	global	 ___gmpn_sub_n
___gmpn_sub_n:
	ld	r6,r3,0			; read first limb from s1_ptr
	extu	r10,r5,3
	ld	r7,r4,0			; read first limb from s2_ptr

	subu	r5,r0,r5
	mak	r5,r5,3<4>
	bcnd.n	eq0,r5,Lzero
	subu.co	r0,r0,r0		; initialize carry

	or	r12,r0,lo16(Lbase)
	or.u	r12,r12,hi16(Lbase)
	addu	r12,r12,r5		; r12 is address for entering in loop

	extu	r5,r5,2			; divide by 4
	subu	r2,r2,r5		; adjust res_ptr
	subu	r3,r3,r5		; adjust s1_ptr
	subu	r4,r4,r5		; adjust s2_ptr

	or	r8,r6,r0

	jmp.n	r12
	 or	r9,r7,r0

Loop:	addu	r3,r3,32
	st	r8,r2,28
	addu	r4,r4,32
	ld	r6,r3,0
	addu	r2,r2,32
	ld	r7,r4,0
Lzero:	subu	r10,r10,1		; subtract 0 + 8r limbs (adj loop cnt)
Lbase:	ld	r8,r3,4
	subu.cio r6,r6,r7
	ld	r9,r4,4
	st	r6,r2,0
	ld	r6,r3,8			; subtract 7 + 8r limbs
	subu.cio r8,r8,r9
	ld	r7,r4,8
	st	r8,r2,4
	ld	r8,r3,12		; subtract 6 + 8r limbs
	subu.cio r6,r6,r7
	ld	r9,r4,12
	st	r6,r2,8
	ld	r6,r3,16		; subtract 5 + 8r limbs
	subu.cio r8,r8,r9
	ld	r7,r4,16
	st	r8,r2,12
	ld	r8,r3,20		; subtract 4 + 8r limbs
	subu.cio r6,r6,r7
	ld	r9,r4,20
	st	r6,r2,16
	ld	r6,r3,24		; subtract 3 + 8r limbs
	subu.cio r8,r8,r9
	ld	r7,r4,24
	st	r8,r2,20
	ld	r8,r3,28		; subtract 2 + 8r limbs
	subu.cio r6,r6,r7
	ld	r9,r4,28
	st	r6,r2,24
	bcnd.n	ne0,r10,Loop		; subtract 1 + 8r limbs
	 subu.cio r8,r8,r9

	st	r8,r2,28		; store most significant limb

	addu.ci r2,r0,r0		; return carry-out from most sign. limb
	jmp.n	 r1
	 xor	r2,r2,1
