; mc88100 __gmpn_mul_1 -- Multiply a limb vector with a single limb and
; store the product in a second limb vector.

; Copyright 1992, 1994, 1995, 2000 Free Software Foundation, Inc.

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
; size		r4
; s2_limb	r5

; Common overhead is about 11 cycles/invocation.

; The speed for S2_LIMB >= 0x10000 is approximately 21 cycles/limb.  (The
; pipeline stalls 2 cycles due to WB contention.)

; The speed for S2_LIMB < 0x10000 is approximately 16 cycles/limb.  (The
; pipeline stalls 2 cycles due to WB contention and 1 cycle due to latency.)

; To enhance speed:
; 1. Unroll main loop 4-8 times.
; 2. Schedule code to avoid WB contention.  It might be tempting to move the
;    ld instruction in the loops down to save 2 cycles (less WB contention),
;    but that looses because the ultimate value will be read from outside
;    the allocated space.  But if we handle the ultimate multiplication in
;    the tail, we can do this.
; 3. Make the multiplication with less instructions.  I think the code for
;    (S2_LIMB >= 0x10000) is not minimal.
; With these techniques the (S2_LIMB >= 0x10000) case would run in 17 or
; less cycles/limb; the (S2_LIMB < 0x10000) case would run in 11
; cycles/limb.  (Assuming infinite unrolling.)

	text
	align	 16
	global	 ___gmpn_mul_1
___gmpn_mul_1:

	; Make S1_PTR and RES_PTR point at the end of their blocks
	; and negate SIZE.
	lda	 r3,r3[r4]
	lda	 r6,r2[r4]	; RES_PTR in r6 since r2 is retval
	subu	 r4,r0,r4

	addu.co	 r2,r0,r0	; r2 = cy = 0
	ld	 r9,r3[r4]
	mask	 r7,r5,0xffff	; r7 = lo(S2_LIMB)
	extu	 r8,r5,16	; r8 = hi(S2_LIMB)
	bcnd.n	 eq0,r8,Lsmall	; jump if (hi(S2_LIMB) == 0)
	 subu	 r6,r6,4

; General code for any value of S2_LIMB.

	; Make a stack frame and save r25 and r26
	subu	 r31,r31,16
	st.d	 r25,r31,8

	; Enter the loop in the middle
	br.n	L1
	addu	 r4,r4,1

Loop:	ld	 r9,r3[r4]
	st	 r26,r6[r4]
; bcnd	ne0,r0,0		; bubble
	addu	 r4,r4,1
L1:	mul	 r26,r9,r5	; low word of product	mul_1	WB ld
	mask	 r12,r9,0xffff	; r12 = lo(s1_limb)	mask_1
	mul	 r11,r12,r7	; r11 =  prod_0		mul_2	WB mask_1
	mul	 r10,r12,r8	; r10 = prod_1a		mul_3
	extu	 r13,r9,16	; r13 = hi(s1_limb)	extu_1	WB mul_1
	mul	 r12,r13,r7	; r12 = prod_1b		mul_4	WB extu_1
	mul	 r25,r13,r8	; r25  = prod_2		mul_5	WB mul_2
	extu	 r11,r11,16	; r11 = hi(prod_0)	extu_2	WB mul_3
	addu	 r10,r10,r11	;			addu_1	WB extu_2
; bcnd	ne0,r0,0		; bubble			WB addu_1
	addu.co	 r10,r10,r12	;				WB mul_4
	mask.u	 r10,r10,0xffff	; move the 16 most significant bits...
	addu.ci	 r10,r10,r0	; ...to the low half of the word...
	rot	 r10,r10,16	; ...and put carry in pos 16.
	addu.co	 r26,r26,r2	; add old carry limb
	bcnd.n	 ne0,r4,Loop
	 addu.ci r2,r25,r10	; compute new carry limb

	st	 r26,r6[r4]
	ld.d	 r25,r31,8
	jmp.n	 r1
	 addu	 r31,r31,16

; Fast code for S2_LIMB < 0x10000
Lsmall:
	; Enter the loop in the middle
	br.n	SL1
	addu	 r4,r4,1

SLoop:	ld	 r9,r3[r4]	;
	st	 r8,r6[r4]	;
	addu	 r4,r4,1	;
SL1:	mul	 r8,r9,r5	; low word of product
	mask	 r12,r9,0xffff	; r12 = lo(s1_limb)
	extu	 r13,r9,16	; r13 = hi(s1_limb)
	mul	 r11,r12,r7	; r11 =  prod_0
	mul	 r12,r13,r7	; r12 = prod_1b
	addu.cio r8,r8,r2	; add old carry limb
	extu	 r10,r11,16	; r11 = hi(prod_0)
	addu	 r10,r10,r12	;
	bcnd.n	 ne0,r4,SLoop
	extu	 r2,r10,16	; r2 = new carry limb

	jmp.n	 r1
	st	 r8,r6[r4]
