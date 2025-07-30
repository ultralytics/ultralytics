dnl  AMD K7 mpn_lshift -- mpn left shift.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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


C K7: 1.21 cycles/limb (at 16 limbs/loop).



dnl  K7: UNROLL_COUNT cycles/limb
dnl           4           1.51
dnl           8           1.26
dnl          16           1.21
dnl          32           1.2
dnl  Maximum possible with the current code is 64.

deflit(UNROLL_COUNT, 16)


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C Shift src,size left by shift many bits and store the result in dst,size.
C Zeros are shifted in at the right.  The bits shifted out at the left are
C the return value.
C
C The comments in mpn_rshift apply here too.

ifdef(`PIC',`
deflit(UNROLL_THRESHOLD, 10)
',`
deflit(UNROLL_THRESHOLD, 10)
')

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)

defframe(SAVE_EDI, -4)
defframe(SAVE_ESI, -8)
defframe(SAVE_EBX, -12)
deflit(SAVE_SIZE, 12)

	TEXT
	ALIGN(32)

PROLOGUE(mpn_lshift)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %eax
	movl	PARAM_SRC, %edx
	subl	$SAVE_SIZE, %esp
deflit(`FRAME',SAVE_SIZE)

	movl	PARAM_SHIFT, %ecx
	movl	%edi, SAVE_EDI

	movl	PARAM_DST, %edi
	decl	%eax
	jnz	L(more_than_one_limb)

	movl	(%edx), %edx

	shldl(	%cl, %edx, %eax)	C eax was decremented to zero

	shll	%cl, %edx

	movl	%edx, (%edi)
	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp

	ret


C -----------------------------------------------------------------------------
L(more_than_one_limb):
	C eax	size-1
	C ebx
	C ecx	shift
	C edx	src
	C esi
	C edi	dst
	C ebp

	movd	PARAM_SHIFT, %mm6
	movd	(%edx,%eax,4), %mm5	C src high limb
	cmp	$UNROLL_THRESHOLD-1, %eax

	jae	L(unroll)
	negl	%ecx
	movd	(%edx), %mm4		C src low limb

	addl	$32, %ecx

	movd	%ecx, %mm7

L(simple_top):
	C eax	loop counter, limbs
	C ebx
	C ecx
	C edx	src
	C esi
	C edi	dst
	C ebp
	C
	C mm0	scratch
	C mm4	src low limb
	C mm5	src high limb
	C mm6	shift
	C mm7	32-shift

	movq	-4(%edx,%eax,4), %mm0
	decl	%eax

	psrlq	%mm7, %mm0

	movd	%mm0, 4(%edi,%eax,4)
	jnz	L(simple_top)


	psllq	%mm6, %mm5
	psllq	%mm6, %mm4

	psrlq	$32, %mm5
	movd	%mm4, (%edi)		C dst low limb

	movd	%mm5, %eax		C return value

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll):
	C eax	size-1
	C ebx	(saved)
	C ecx	shift
	C edx	src
	C esi
	C edi	dst
	C ebp
	C
	C mm5	src high limb, for return value
	C mm6	lshift

	movl	%esi, SAVE_ESI
	movl	%ebx, SAVE_EBX
	leal	-4(%edx,%eax,4), %edx   C &src[size-2]

	testb	$4, %dl
	movq	(%edx), %mm1		C src high qword

	jz	L(start_src_aligned)


	C src isn't aligned, process high limb (marked xxx) separately to
	C make it so
	C
	C  source    -4(edx,%eax,4)
	C                  |
	C  +-------+-------+-------+--
	C  |  xxx          |
	C  +-------+-------+-------+--
	C        0mod8   4mod8   0mod8
	C
	C  dest      -4(edi,%eax,4)
	C                  |
	C  +-------+-------+--
	C  |  xxx  |       |
	C  +-------+-------+--

	psllq	%mm6, %mm1
	subl	$4, %edx
	movl	%eax, PARAM_SIZE	C size-1

	psrlq	$32, %mm1
	decl	%eax			C size-2 is new size-1

	movd	%mm1, 4(%edi,%eax,4)
	movq	(%edx), %mm1		C new src high qword
L(start_src_aligned):


	leal	-4(%edi,%eax,4), %edi   C &dst[size-2]
	psllq	%mm6, %mm5

	testl	$4, %edi
	psrlq	$32, %mm5		C return value

	jz	L(start_dst_aligned)


	C dst isn't aligned, subtract 4 bytes to make it so, and pretend the
	C shift is 32 bits extra.  High limb of dst (marked xxx) handled
	C here separately.
	C
	C  source       %edx
	C  +-------+-------+--
	C  |      mm1      |
	C  +-------+-------+--
	C                0mod8   4mod8
	C
	C  dest         %edi
	C  +-------+-------+-------+--
	C  |  xxx  |
	C  +-------+-------+-------+--
	C        0mod8   4mod8   0mod8

	movq	%mm1, %mm0
	psllq	%mm6, %mm1
	addl	$32, %ecx		C shift+32

	psrlq	$32, %mm1

	movd	%mm1, 4(%edi)
	movq	%mm0, %mm1
	subl	$4, %edi

	movd	%ecx, %mm6		C new lshift
L(start_dst_aligned):

	decl	%eax			C size-2, two last limbs handled at end
	movq	%mm1, %mm2		C copy of src high qword
	negl	%ecx

	andl	$-2, %eax		C round size down to even
	addl	$64, %ecx

	movl	%eax, %ebx
	negl	%eax

	andl	$UNROLL_MASK, %eax
	decl	%ebx

	shll	%eax

	movd	%ecx, %mm7		C rshift = 64-lshift

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(entry) (%eax,%eax,4), %esi
')
	shrl	$UNROLL_LOG2, %ebx	C loop counter

	leal	ifelse(UNROLL_BYTES,256,128) -8(%edx,%eax,2), %edx
	leal	ifelse(UNROLL_BYTES,256,128) (%edi,%eax,2), %edi
	movl	PARAM_SIZE, %eax	C for use at end
	jmp	*%esi


ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	leal	(%eax,%eax,4), %esi
	addl	$L(entry)-L(here), %esi
	addl	(%esp), %esi

	ret_internal
')


C -----------------------------------------------------------------------------
	ALIGN(32)
L(top):
	C eax	size (for use at end)
	C ebx	loop counter
	C ecx	rshift
	C edx	src
	C esi	computed jump
	C edi	dst
	C ebp
	C
	C mm0	scratch
	C mm1	\ carry (alternating, mm2 first)
	C mm2	/
	C mm6	lshift
	C mm7	rshift
	C
	C 10 code bytes/limb
	C
	C The two chunks differ in whether mm1 or mm2 hold the carry.
	C The computed jump puts the initial carry in both mm1 and mm2.

L(entry):
deflit(CHUNK_COUNT, 4)
forloop(i, 0, UNROLL_COUNT/CHUNK_COUNT-1, `
	deflit(`disp0', eval(-i*CHUNK_COUNT*4 ifelse(UNROLL_BYTES,256,-128)))
	deflit(`disp1', eval(disp0 - 8))

Zdisp(	movq,	disp0,(%edx), %mm0)
	psllq	%mm6, %mm2

	movq	%mm0, %mm1
	psrlq	%mm7, %mm0

	por	%mm2, %mm0
Zdisp(	movq,	%mm0, disp0,(%edi))


Zdisp(	movq,	disp1,(%edx), %mm0)
	psllq	%mm6, %mm1

	movq	%mm0, %mm2
	psrlq	%mm7, %mm0

	por	%mm1, %mm0
Zdisp(	movq,	%mm0, disp1,(%edi))
')

	subl	$UNROLL_BYTES, %edx
	subl	$UNROLL_BYTES, %edi
	decl	%ebx

	jns	L(top)



define(`disp', `m4_empty_if_zero(eval($1 ifelse(UNROLL_BYTES,256,-128)))')

L(end):
	testb	$1, %al
	movl	SAVE_EBX, %ebx
	psllq	%mm6, %mm2	C wanted left shifted in all cases below

	movd	%mm5, %eax

	movl	SAVE_ESI, %esi
	jz	L(end_even)


L(end_odd):

	C Size odd, destination was aligned.
	C
	C                 source        edx+8   edx+4
	C                 --+---------------+-------+
	C                   |      mm2      |       |
	C                 --+---------------+-------+
	C
	C dest                            edi
	C --+---------------+---------------+-------+
	C   |   written     |               |       |
	C --+---------------+---------------+-------+
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C Size odd, destination was unaligned.
	C
	C                 source        edx+8   edx+4
	C                 --+---------------+-------+
	C                   |      mm2      |       |
	C                 --+---------------+-------+
	C
	C         dest                            edi
	C         --+---------------+---------------+
	C           |   written     |               |
	C         --+---------------+---------------+
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C In both cases there's one extra limb of src to fetch and combine
	C with mm2 to make a qword at (%edi), and in the aligned case
	C there's an extra limb of dst to be formed from that extra src limb
	C left shifted.

	movd	disp(4) (%edx), %mm0
	testb	$32, %cl

	movq	%mm0, %mm1
	psllq	$32, %mm0

	psrlq	%mm7, %mm0
	psllq	%mm6, %mm1

	por	%mm2, %mm0

	movq	%mm0, disp(0) (%edi)
	jz	L(end_odd_unaligned)
	movd	%mm1, disp(-4) (%edi)
L(end_odd_unaligned):

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret


L(end_even):

	C Size even, destination was aligned.
	C
	C                 source        edx+8
	C                 --+---------------+
	C                   |      mm2      |
	C                 --+---------------+
	C
	C dest                            edi
	C --+---------------+---------------+
	C   |   written     |               |
	C --+---------------+---------------+
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C Size even, destination was unaligned.
	C
	C               source          edx+8
	C                 --+---------------+
	C                   |      mm2      |
	C                 --+---------------+
	C
	C         dest                  edi+4
	C         --+---------------+-------+
	C           |    written    |       |
	C         --+---------------+-------+
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C The movq for the aligned case overwrites the movd for the
	C unaligned case.

	movq	%mm2, %mm0
	psrlq	$32, %mm2

	testb	$32, %cl
	movd	%mm2, disp(4) (%edi)

	jz	L(end_even_unaligned)
	movq	%mm0, disp(0) (%edi)
L(end_even_unaligned):

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret

EPILOGUE()
