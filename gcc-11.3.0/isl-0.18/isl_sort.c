/*
 * The code of this file was taken from http://jeffreystedfast.blogspot.be,
 * where it was posted in 2011 by Jeffrey Stedfast under the MIT license.
 * The MIT license text is as follows:
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <isl_sort.h>

#define MID(lo, hi) (lo + ((hi - lo) >> 1))

/* The code here is an optimized merge sort. Starting from a generic merge sort
 * the following optimizations were applied:
 *
 * o Batching of memcpy() calls: Instead of calling memcpy() to copy each and
 *   every element into a temporary buffer, blocks of elements are copied
 *   at a time.
 *
 * o To reduce the number of memcpy() calls further, copying leading
 *   and trailing elements into our temporary buffer is avoided, in case it is
 *   not necessary to merge them.
 *
 * A further optimization could be to specialize memcpy calls based on the
 * size of the types we compare. For now, this code does not include the
 * relevant optimization, as clang e.g. inlines a very efficient memcpy()
 * implementation. It is not clear, that the specialized version as provided in
 * the blog post, is really superior to the one that will be inlined by
 * default. So we decided to keep the code simple until this optimization was
 * proven to be beneficial.
 */

static void
msort (void *array, void *buf, size_t low, size_t high, size_t size,
       int (* compare) (const void *, const void *, void *), void *arg)
{
    char *a1, *al, *am, *ah, *ls, *hs, *lo, *hi, *b;
    size_t copied = 0;
    size_t mid;

    mid = MID (low, high);

    if (mid + 1 < high)
        msort (array, buf, mid + 1, high, size, compare, arg);

    if (mid > low)
        msort (array, buf, low, mid, size, compare, arg);

    ah = ((char *) array) + ((high + 1) * size);
    am = ((char *) array) + ((mid + 1) * size);
    a1 = al = ((char *) array) + (low * size);

    b = (char *) buf;
    lo = al;
    hi = am;

    do {
        ls = lo;
        hs = hi;

        if (lo > al || hi > am) {
            /* our last loop already compared lo & hi and found lo <= hi */
            lo += size;
        }

        while (lo < am && compare (lo, hi, arg) <= 0)
            lo += size;

        if (lo < am) {
            if (copied == 0) {
                /* avoid copying the leading items */
                a1 = lo;
                ls = lo;
            }

            /* our last compare tells us hi < lo */
            hi += size;

            while (hi < ah && compare (hi, lo, arg) < 0)
                hi += size;

            if (lo > ls) {
                memcpy (b, ls, lo - ls);
                copied += (lo - ls);
                b += (lo - ls);
            }

            memcpy (b, hs, hi - hs);
            copied += (hi - hs);
            b += (hi - hs);
        } else if (copied) {
            memcpy (b, ls, lo - ls);
            copied += (lo - ls);
            b += (lo - ls);

            /* copy everything we needed to re-order back into array */
            memcpy (a1, buf, copied);
            return;
        } else {
            /* everything already in order */
            return;
        }
    } while (hi < ah);

    if (lo < am) {
        memcpy (b, lo, am - lo);
        copied += (am - lo);
    }

    memcpy (a1, buf, copied);
}

static int
MergeSort (void *base, size_t nmemb, size_t size,
           int (* compare) (const void *, const void *, void *), void *arg)
{
    void *tmp;

    if (nmemb < 2)
        return 0;

    if (!(tmp = malloc (nmemb * size))) {
        errno = ENOMEM;
        return -1;
    }

    msort (base, tmp, 0, nmemb - 1, size, compare, arg);

    free (tmp);

    return 0;
}

int isl_sort(void *const pbase, size_t total_elems, size_t size,
	int (*cmp)(const void *, const void *, void *arg), void *arg)
{
    return MergeSort (pbase, total_elems, size, cmp, arg);
}
