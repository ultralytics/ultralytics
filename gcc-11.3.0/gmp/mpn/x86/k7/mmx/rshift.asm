dnl  AMD K7 mpn_rshift -- mpn right shift.

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


C mp_limb_t mpn_rshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C Shift src,size right by shift many bits and store the result in dst,size.
C Zeros are shifted in at the left.  The bits shifted out at the right are
C the return value.
C
C This code uses 64-bit MMX operations, which makes it possible to handle
C two limbs at a time, for a theoretical 1.0 cycles/limb.  Plain integer
C code, on the other hand, suffers from shrd being a vector path decode and
C running at 3 cycles back-to-back.
C
C Full speed depends on source and destination being aligned, and some hairy
C setups and finish-ups are done to arrange this for the loop.

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

PROLOGUE(mpn_rshift)
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

	movl	(%edx), %edx		C src limb

	shrdl(	%cl, %edx, %eax)	C eax was decremented to zero

	shrl	%cl, %edx

	movl	%edx, (%edi)		C dst limb
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

	movd	PARAM_SHIFT, %mm6	C rshift
	movd	(%edx), %mm5		C src low limb
	cmp	$UNROLL_THRESHOLD-1, %eax

	jae	L(unroll)
	leal	(%edx,%eax,4), %edx	C &src[size-1]
	leal	-4(%edi,%eax,4), %edi	C &dst[size-2]

	movd	(%edx), %mm4		C src high limb
	negl	%eax


L(simple_top):
	C eax	loop counter, limbs, negative
	C ebx
	C ecx	shift
	C edx	carry
	C edx	&src[size-1]
	C edi	&dst[size-2]
	C ebp
	C
	C mm0	scratch
	C mm4	src high limb
	C mm5	src low limb
	C mm6	shift

	movq	(%edx,%eax,4), %mm0
	incl	%eax

	psrlq	%mm6, %mm0

	movd	%mm0, (%edi,%eax,4)
	jnz	L(simple_top)


	psllq	$32, %mm5
	psrlq	%mm6, %mm4

	psrlq	%mm6, %mm5
	movd	%mm4, 4(%edi)		C dst high limb

	movd	%mm5, %eax		C return value

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll):
	C eax	size-1
	C ebx
	C ecx	shift
	C edx	src
	C esi
	C edi	dst
	C ebp
	C
	C mm5	src low limb
	C mm6	rshift

	testb	$4, %dl
	movl	%esi, SAVE_ESI
	movl	%ebx, SAVE_EBX

	psllq	$32, %mm5
	jz	L(start_src_aligned)


	C src isn't aligned, process low limb separately (marked xxx) and
	C step src and dst by one limb, making src aligned.
	C
	C source                  edx
	C --+-------+-------+-------+
	C           |          xxx  |
	C --+-------+-------+-------+
	C         4mod8   0mod8   4mod8
	C
	C         dest            edi
	C         --+-------+-------+
	C           |       |  xxx  |
	C         --+-------+-------+

	movq	(%edx), %mm0		C src low two limbs
	addl	$4, %edx
	movl	%eax, PARAM_SIZE	C size-1

	addl	$4, %edi
	decl	%eax			C size-2 is new size-1

	psrlq	%mm6, %mm0
	movl	%edi, PARAM_DST		C new dst

	movd	%mm0, -4(%edi)
L(start_src_aligned):


	movq	(%edx), %mm1		C src low two limbs
	decl	%eax			C size-2, two last limbs handled at end
	testl	$4, %edi

	psrlq	%mm6, %mm5
	jz	L(start_dst_aligned)


	C dst isn't aligned, add 4 to make it so, and pretend the shift is
	C 32 bits extra.  Low limb of dst (marked xxx) handled here separately.
	C
	C          source          edx
	C          --+-------+-------+
	C            |      mm1      |
	C          --+-------+-------+
	C                  4mod8   0mod8
	C
	C  dest                    edi
	C  --+-------+-------+-------+
	C                    |  xxx  |
	C  --+-------+-------+-------+
	C          4mod8   0mod8   4mod8

	movq	%mm1, %mm0
	psrlq	%mm6, %mm1
	addl	$32, %ecx		C shift+32

	movd	%mm1, (%edi)
	movq	%mm0, %mm1
	addl	$4, %edi		C new dst

	movd	%ecx, %mm6
L(start_dst_aligned):


	movq	%mm1, %mm2		C copy of src low two limbs
	negl	%ecx
	andl	$-2, %eax		C round size down to even

	movl	%eax, %ebx
	negl	%eax
	addl	$64, %ecx

	andl	$UNROLL_MASK, %eax
	decl	%ebx

	shll	%eax

	movd	%ecx, %mm7		C lshift = 64-rshift

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(entry) (%eax,%eax,4), %esi
	negl	%eax
')
	shrl	$UNROLL_LOG2, %ebx	C loop counter

	leal	ifelse(UNROLL_BYTES,256,128+) 8(%edx,%eax,2), %edx
	leal	ifelse(UNROLL_BYTES,256,128) (%edi,%eax,2), %edi
	movl	PARAM_SIZE, %eax	C for use at end

	jmp	*%esi


ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	leal	(%eax,%eax,4), %esi
	addl	$L(entry)-L(here), %esi
	addl	(%esp), %esi
	negl	%eax

	ret_internal
')


C -----------------------------------------------------------------------------
	ALIGN(64)
L(top):
	C eax	size, for use at end
	C ebx	loop counter
	C ecx	lshift
	C edx	src
	C esi	was computed jump
	C edi	dst
	C ebp
	C
	C mm0	scratch
	C mm1	\ carry (alternating)
	C mm2	/
	C mm6	rshift
	C mm7	lshift
	C
	C 10 code bytes/limb
	C
	C The two chunks differ in whether mm1 or mm2 hold the carry.
	C The computed jump puts the initial carry in both mm1 and mm2.

L(entry):
deflit(CHUNK_COUNT, 4)
forloop(i, 0, UNROLL_COUNT/CHUNK_COUNT-1, `
	deflit(`disp0', eval(i*CHUNK_COUNT*4 ifelse(UNROLL_BYTES,256,-128)))
	deflit(`disp1', eval(disp0 + 8))

Zdisp(	movq,	disp0,(%edx), %mm0)
	psrlq	%mm6, %mm2

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	por	%mm2, %mm0
Zdisp(	movq,	%mm0, disp0,(%edi))


Zdisp(	movq,	disp1,(%edx), %mm0)
	psrlq	%mm6, %mm1

	movq	%mm0, %mm2
	psllq	%mm7, %mm0

	por	%mm1, %mm0
Zdisp(	movq,	%mm0, disp1,(%edi))
')

	addl	$UNROLL_BYTES, %edx
	addl	$UNROLL_BYTES, %edi
	decl	%ebx

	jns	L(top)


deflit(`disp0', ifelse(UNROLL_BYTES,256,-128))
deflit(`disp1', eval(disp0-0 + 8))

	testb	$1, %al
	psrlq	%mm6, %mm2	C wanted rshifted in all cases below
	movl	SAVE_ESI, %esi

	movd	%mm5, %eax		C return value

	movl	SAVE_EBX, %ebx
	jz	L(end_even)


	C Size odd, destination was aligned.
	C
	C source
	C       edx
	C +-------+---------------+--
	C |       |      mm2      |
	C +-------+---------------+--
	C
	C dest                  edi
	C +-------+---------------+---------------+--
	C |       |               |    written    |
	C +-------+---------------+---------------+--
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C Size odd, destination was unaligned.
	C
	C source
	C       edx
	C +-------+---------------+--
	C |       |      mm2      |
	C +-------+---------------+--
	C
	C dest          edi
	C +---------------+---------------+--
	C |               |    written    |
	C +---------------+---------------+--
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C In both cases there's one extra limb of src to fetch and combine
	C with mm2 to make a qword to store, and in the aligned case there's
	C a further extra limb of dst to be formed.


	movd	disp0(%edx), %mm0
	movq	%mm0, %mm1

	psllq	%mm7, %mm0
	testb	$32, %cl

	por	%mm2, %mm0
	psrlq	%mm6, %mm1

	movq	%mm0, disp0(%edi)
	jz	L(finish_odd_unaligned)

	movd	%mm1, disp1(%edi)
L(finish_odd_unaligned):

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret


L(end_even):

	C Size even, destination was aligned.
	C
	C source
	C +---------------+--
	C |      mm2      |
	C +---------------+--
	C
	C dest          edi
	C +---------------+---------------+--
	C |               |      mm3      |
	C +---------------+---------------+--
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C Size even, destination was unaligned.
	C
	C source
	C +---------------+--
	C |      mm2      |
	C +---------------+--
	C
	C dest  edi
	C +-------+---------------+--
	C |       |      mm3      |
	C +-------+---------------+--
	C
	C mm6 = shift+32
	C mm7 = 64-(shift+32)


	C The movd for the unaligned case is the same data as the movq for
	C the aligned case, it's just a choice between whether one or two
	C limbs should be written.


	testb	$32, %cl
	movd	%mm2, disp0(%edi)

	jz	L(end_even_unaligned)

	movq	%mm2, disp0(%edi)
L(end_even_unaligned):

	movl	SAVE_EDI, %edi
	addl	$SAVE_SIZE, %esp
	emms

	ret

EPILOGUE()
