/* Time routines for speed measurements.

Copyright 1999-2004, 2010-2012 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


/* Usage:

   The code in this file implements the lowest level of time measuring,
   simple one-time measuring of time between two points.

   void speed_starttime (void)
   double speed_endtime (void)
       Call speed_starttime to start measuring, and then call speed_endtime
       when done.

       speed_endtime returns the time taken, in seconds.  Or if the timebase
       is in CPU cycles and the CPU frequency is unknown then speed_endtime
       returns cycles.  Applications can identify the cycles return by
       checking for speed_cycletime (described below) equal to 1.0.

       If some sort of temporary glitch occurs then speed_endtime returns
       0.0.  Currently this is for various cases where a negative time has
       occurred.  This unfortunately occurs with getrusage on some systems,
       and with the hppa cycle counter on hpux.

   double speed_cycletime
       The time in seconds for each CPU cycle.  For example on a 100 MHz CPU
       this would be 1.0e-8.

       If the CPU frequency is unknown, then speed_cycletime is either 0.0
       or 1.0.  It's 0.0 when speed_endtime is returning seconds, or it's
       1.0 when speed_endtime is returning cycles.

       It may be noted that "speed_endtime() / speed_cycletime" gives a
       measured time in cycles, irrespective of whether speed_endtime is
       returning cycles or seconds.  (Assuming cycles can be had, ie. it's
       either cycles already or the cpu frequency is known.  See also
       speed_cycletime_need_cycles below.)

   double speed_unittime
       The unit of time measurement accuracy for the timing method in use.
       This is in seconds or cycles, as per speed_endtime.

   char speed_time_string[]
       A null-terminated string describing the time method in use.

   void speed_time_init (void)
       Initialize time measuring.  speed_starttime() does this
       automatically, so it's only needed if an application wants to inspect
       the above global variables before making a measurement.

   int speed_precision
       The intended accuracy of time measurements.  speed_measure() in
       common.c for instance runs target routines with enough repetitions so
       it takes at least "speed_unittime * speed_precision" (this expression
       works for both cycles or seconds from speed_endtime).

       A program can provide an option so the user to set speed_precision.
       If speed_precision is zero when speed_time_init or speed_starttime
       first run then it gets a default based on the measuring method
       chosen.  (More precision for higher accuracy methods.)

   void speed_cycletime_need_seconds (void)
       Call this to demand that speed_endtime will return seconds, and not
       cycles.  If only cycles are available then an error is printed and
       the program exits.

   void speed_cycletime_need_cycles (void)
       Call this to demand that speed_cycletime is non-zero, so that
       "speed_endtime() / speed_cycletime" will give times in cycles.



   Notes:

   Various combinations of cycle counter, read_real_time(), getrusage(),
   gettimeofday() and times() can arise, according to which are available
   and their precision.


   Allowing speed_endtime() to return either seconds or cycles is only a
   slight complication and makes it possible for the speed program to do
   some sensible things without demanding the CPU frequency.  If seconds are
   being measured then it can always print seconds, and if cycles are being
   measured then it can always print them without needing to know how long
   they are.  Also the tune program doesn't care at all what the units are.

   GMP_CPU_FREQUENCY can always be set when the automated methods in freq.c
   fail.  This will be needed if times in seconds are wanted but a cycle
   counter is being used, or if times in cycles are wanted but getrusage or
   another seconds based timer is in use.

   If the measuring method uses a cycle counter but supplements it with
   getrusage or the like, then knowing the CPU frequency is mandatory since
   the code compares values from the two.


   Not done:

   Solaris gethrtime() seems no more than a slow way to access the Sparc V9
   cycle counter.  gethrvtime() seems to be relevant only to light weight
   processes, it doesn't for instance give nanosecond virtual time.  So
   neither of these are used.


   Bugs:

   getrusage_microseconds_p is fundamentally flawed, getrusage and
   gettimeofday can have resolutions other than clock ticks or microseconds,
   for instance IRIX 5 has a tick of 10 ms but a getrusage of 1 ms.


   Enhancements:

   The SGI hardware counter has 64 bits on some machines, which could be
   used when available.  But perhaps 32 bits is enough range, and then rely
   on the getrusage supplement.

   Maybe getrusage (or times) should be used as a supplement for any
   wall-clock measuring method.  Currently a wall clock with a good range
   (eg. a 64-bit cycle counter) is used without a supplement.

   On PowerPC the timebase registers could be used, but would have to do
   something to find out the speed.  On 6xx chips it's normally 1/4 bus
   speed, on 4xx chips it's either that or an external clock.  Measuring
   against gettimeofday might be ok.  */


#include "config.h"

#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h> /* for getenv() */

#if HAVE_FCNTL_H
#include <fcntl.h>  /* for open() */
#endif

#if HAVE_STDINT_H
#include <stdint.h> /* for uint64_t */
#endif

#if HAVE_UNISTD_H
#include <unistd.h> /* for sysconf() */
#endif

#include <sys/types.h>

#if TIME_WITH_SYS_TIME
# include <sys/time.h>  /* for struct timeval */
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#if HAVE_SYS_MMAN_H
#include <sys/mman.h>      /* for mmap() */
#endif

#if HAVE_SYS_RESOURCE_H
#include <sys/resource.h>  /* for struct rusage */
#endif

#if HAVE_SYS_SYSSGI_H
#include <sys/syssgi.h>    /* for syssgi() */
#endif

#if HAVE_SYS_SYSTEMCFG_H
#include <sys/systemcfg.h> /* for RTC_POWER on AIX */
#endif

#if HAVE_SYS_TIMES_H
#include <sys/times.h>  /* for times() and struct tms */
#endif

#include "gmp.h"
#include "gmp-impl.h"

#include "speed.h"


/* strerror is only used for some stuff on newish systems, no need to have a
   proper replacement */
#if ! HAVE_STRERROR
#define strerror(n)  "<strerror not available>"
#endif


char    speed_time_string[256];
int     speed_precision = 0;
double  speed_unittime;
double  speed_cycletime = 0.0;


/* don't rely on "unsigned" to "double" conversion, it's broken in SunOS 4
   native cc */
#define M_2POWU   (((double) INT_MAX + 1.0) * 2.0)

#define M_2POW32  4294967296.0
#define M_2POW64  (M_2POW32 * M_2POW32)


/* Conditionals for the time functions available are done with normal C
   code, which is a lot easier than wildly nested preprocessor directives.

   The choice of what to use is partly made at run-time, according to
   whether the cycle counter works and the measured accuracy of getrusage
   and gettimeofday.

   A routine that's not available won't be getting called, but is an abort()
   to be sure it isn't called mistakenly.

   It can be assumed that if a function exists then its data type will, but
   if the function doesn't then the data type might or might not exist, so
   the type can't be used unconditionally.  The "struct_rusage" etc macros
   provide dummies when the respective function doesn't exist. */


#if HAVE_SPEED_CYCLECOUNTER
static const int have_cycles = HAVE_SPEED_CYCLECOUNTER;
#else
static const int have_cycles = 0;
#define speed_cyclecounter(p)  ASSERT_FAIL (speed_cyclecounter not available)
#endif

/* "stck" returns ticks since 1 Jan 1900 00:00 GMT, where each tick is 2^-12
   microseconds.  Same #ifdefs here as in longlong.h.  */
#if defined (__GNUC__) && ! defined (NO_ASM)                            \
  && (defined (__i370__) || defined (__s390__) || defined (__mvs__))
static const int  have_stck = 1;
static const int  use_stck = 1;  /* always use when available */
typedef uint64_t  stck_t; /* gcc for s390 is quite new, always has uint64_t */
#define STCK(timestamp)                 \
  do {                                  \
    asm ("stck %0" : "=Q" (timestamp)); \
  } while (0)
#else
static const int  have_stck = 0;
static const int  use_stck = 0;
typedef unsigned long  stck_t;   /* dummy */
#define STCK(timestamp)  ASSERT_FAIL (stck instruction not available)
#endif
#define STCK_PERIOD      (1.0 / 4096e6)   /* 2^-12 microseconds */

/* mftb
   Enhancement: On 64-bit chips mftb gives a 64-bit value, no need for mftbu
   and a loop (see powerpc64.asm).  */
#if HAVE_HOST_CPU_FAMILY_powerpc
static const int  have_mftb = 1;
#if defined (__GNUC__) && ! defined (NO_ASM)
#define MFTB(a)                         \
  do {                                  \
    unsigned  __h1, __l, __h2;          \
    do {                                \
      asm volatile ("mftbu %0\n"        \
		    "mftb  %1\n"        \
		    "mftbu %2"          \
		    : "=r" (__h1),      \
		      "=r" (__l),       \
		      "=r" (__h2));     \
    } while (__h1 != __h2);             \
    a[0] = __l;                         \
    a[1] = __h1;                        \
  } while (0)
#else
#define MFTB(a)   mftb_function (a)
#endif
#else /* ! powerpc */
static const int  have_mftb = 0;
#define MFTB(a)                         \
  do {                                  \
    a[0] = 0;                           \
    a[1] = 0;                           \
    ASSERT_FAIL (mftb not available);   \
  } while (0)
#endif

/* Unicos 10.X has syssgi(), but not mmap(). */
#if HAVE_SYSSGI && HAVE_MMAP
static const int  have_sgi = 1;
#else
static const int  have_sgi = 0;
#endif

#if HAVE_READ_REAL_TIME
static const int have_rrt = 1;
#else
static const int have_rrt = 0;
#define read_real_time(t,s)     ASSERT_FAIL (read_real_time not available)
#define time_base_to_time(t,s)  ASSERT_FAIL (time_base_to_time not available)
#define RTC_POWER     1
#define RTC_POWER_PC  2
#define timebasestruct_t   struct timebasestruct_dummy
struct timebasestruct_dummy {
  int             flag;
  unsigned int    tb_high;
  unsigned int    tb_low;
};
#endif

#if HAVE_CLOCK_GETTIME
static const int have_cgt = 1;
#define struct_timespec  struct timespec
#else
static const int have_cgt = 0;
#define struct_timespec       struct timespec_dummy
#define clock_gettime(id,ts)  (ASSERT_FAIL (clock_gettime not available), -1)
#define clock_getres(id,ts)   (ASSERT_FAIL (clock_getres not available), -1)
#endif

#if HAVE_GETRUSAGE
static const int have_grus = 1;
#define struct_rusage   struct rusage
#else
static const int have_grus = 0;
#define getrusage(n,ru)  ASSERT_FAIL (getrusage not available)
#define struct_rusage    struct rusage_dummy
#endif

#if HAVE_GETTIMEOFDAY
static const int have_gtod = 1;
#define struct_timeval   struct timeval
#else
static const int have_gtod = 0;
#define gettimeofday(tv,tz)  ASSERT_FAIL (gettimeofday not available)
#define struct_timeval   struct timeval_dummy
#endif

#if HAVE_TIMES
static const int have_times = 1;
#define struct_tms   struct tms
#else
static const int have_times = 0;
#define times(tms)   ASSERT_FAIL (times not available)
#define struct_tms   struct tms_dummy
#endif

struct tms_dummy {
  long  tms_utime;
};
struct timeval_dummy {
  long  tv_sec;
  long  tv_usec;
};
struct rusage_dummy {
  struct_timeval ru_utime;
};
struct timespec_dummy {
  long  tv_sec;
  long  tv_nsec;
};

static int  use_cycles;
static int  use_mftb;
static int  use_sgi;
static int  use_rrt;
static int  use_cgt;
static int  use_gtod;
static int  use_grus;
static int  use_times;
static int  use_tick_boundary;

static unsigned         start_cycles[2];
static stck_t           start_stck;
static unsigned         start_mftb[2];
static unsigned         start_sgi;
static timebasestruct_t start_rrt;
static struct_timespec  start_cgt;
static struct_rusage    start_grus;
static struct_timeval   start_gtod;
static struct_tms       start_times;

static double  cycles_limit = 1e100;
static double  mftb_unittime;
static double  sgi_unittime;
static double  cgt_unittime;
static double  grus_unittime;
static double  gtod_unittime;
static double  times_unittime;

/* for RTC_POWER format, ie. seconds and nanoseconds */
#define TIMEBASESTRUCT_SECS(t)  ((t)->tb_high + (t)->tb_low * 1e-9)


/* Return a string representing a time in seconds, nicely formatted.
   Eg. "10.25ms".  */
char *
unittime_string (double t)
{
  static char  buf[128];

  const char  *unit;
  int         prec;

  /* choose units and scale */
  if (t < 1e-6)
    t *= 1e9, unit = "ns";
  else if (t < 1e-3)
    t *= 1e6, unit = "us";
  else if (t < 1.0)
    t *= 1e3, unit = "ms";
  else
    unit = "s";

  /* want 4 significant figures */
  if (t < 1.0)
    prec = 4;
  else if (t < 10.0)
    prec = 3;
  else if (t < 100.0)
    prec = 2;
  else
    prec = 1;

  sprintf (buf, "%.*f%s", prec, t, unit);
  return buf;
}


static jmp_buf  cycles_works_buf;

static RETSIGTYPE
cycles_works_handler (int sig)
{
  longjmp (cycles_works_buf, 1);
}

int
cycles_works_p (void)
{
  static int  result = -1;

  if (result != -1)
    goto done;

  /* FIXME: On linux, the cycle counter is not saved and restored over
   * context switches, making it almost useless for precise cputime
   * measurements. When available, it's better to use clock_gettime,
   * which seems to have reasonable accuracy (tested on x86_32,
   * linux-2.6.26, glibc-2.7). However, there are also some linux
   * systems where clock_gettime is broken in one way or the other,
   * like CLOCK_PROCESS_CPUTIME_ID not implemented (easy case) or
   * kind-of implemented but broken (needs code to detect that), and
   * on those systems a wall-clock cycle counter is the least bad
   * fallback.
   *
   * So we need some code to disable the cycle counter on some but not
   * all linux systems. */
#ifdef SIGILL
  {
    RETSIGTYPE (*old_handler) (int);
    unsigned  cycles[2];

    old_handler = signal (SIGILL, cycles_works_handler);
    if (old_handler == SIG_ERR)
      {
	if (speed_option_verbose)
	  printf ("cycles_works_p(): SIGILL not supported, assuming speed_cyclecounter() works\n");
	goto yes;
      }
    if (setjmp (cycles_works_buf))
      {
	if (speed_option_verbose)
	  printf ("cycles_works_p(): SIGILL during speed_cyclecounter(), so doesn't work\n");
	result = 0;
	goto done;
      }
    speed_cyclecounter (cycles);
    signal (SIGILL, old_handler);
    if (speed_option_verbose)
      printf ("cycles_works_p(): speed_cyclecounter() works\n");
  }
#else

  if (speed_option_verbose)
    printf ("cycles_works_p(): SIGILL not defined, assuming speed_cyclecounter() works\n");
  goto yes;
#endif

 yes:
  result = 1;

 done:
  return result;
}


/* The number of clock ticks per second, but looking at sysconf rather than
   just CLK_TCK, where possible.  */
long
clk_tck (void)
{
  static long  result = -1L;
  if (result != -1L)
    return result;

#if HAVE_SYSCONF
  result = sysconf (_SC_CLK_TCK);
  if (result != -1L)
    {
      if (speed_option_verbose)
	printf ("sysconf(_SC_CLK_TCK) is %ld per second\n", result);
      return result;
    }

  fprintf (stderr,
	   "sysconf(_SC_CLK_TCK) not working, using CLK_TCK instead\n");
#endif

#ifdef CLK_TCK
  result = CLK_TCK;
  if (speed_option_verbose)
    printf ("CLK_TCK is %ld per second\n", result);
  return result;
#else
  fprintf (stderr, "CLK_TCK not defined, cannot continue\n");
  abort ();
#endif
}


/* If two times can be observed less than half a clock tick apart, then
   assume "get" is microsecond accurate.

   Two times only 1 microsecond apart are not believed, since some kernels
   take it upon themselves to ensure gettimeofday doesn't return the same
   value twice, for the benefit of applications using it for a timestamp.
   This is obviously very stupid given the speed of CPUs these days.

   Making "reps" many calls to noop_1() is designed to waste some CPU, with
   a view to getting measurements 2 microseconds (or more) apart.  "reps" is
   increased progressively until such a period is seen.

   The outer loop "attempts" are just to allow for any random nonsense or
   system load upsetting the measurements (ie. making two successive calls
   to "get" come out as a longer interval than normal).

   Bugs:

   The assumption that any interval less than a half tick implies
   microsecond resolution is obviously fairly rash, the true resolution
   could be anything between a microsecond and that half tick.  Perhaps
   something special would have to be done on a system where this is the
   case, since there's no obvious reliable way to detect it
   automatically.  */

#define MICROSECONDS_P(name, type, get, sec, usec)                      \
  {                                                                     \
    static int  result = -1;                                            \
    type      st, et;                                                   \
    long      dt, half_tick;                                            \
    unsigned  attempt, reps, i, j;                                      \
									\
    if (result != -1)                                                   \
      return result;                                                    \
									\
    result = 0;                                                         \
    half_tick = (1000000L / clk_tck ()) / 2;                            \
									\
    for (attempt = 0; attempt < 5; attempt++)                           \
      {                                                                 \
	reps = 0;                                                       \
	for (;;)                                                        \
	  {                                                             \
	    get (st);                                                   \
	    for (i = 0; i < reps; i++)                                  \
	      for (j = 0; j < 100; j++)                                 \
		noop_1 (CNST_LIMB(0));                                  \
	    get (et);                                                   \
									\
	    dt = (sec(et)-sec(st))*1000000L + usec(et)-usec(st);        \
									\
	    if (speed_option_verbose >= 2)                              \
	      printf ("%s attempt=%u, reps=%u, dt=%ld\n",               \
		      name, attempt, reps, dt);                         \
									\
	    if (dt >= 2)                                                \
	      break;                                                    \
									\
	    reps = (reps == 0 ? 1 : 2*reps);                            \
	    if (reps == 0)                                              \
	      break;  /* uint overflow, not normal */                   \
	  }                                                             \
									\
	if (dt < half_tick)                                             \
	  {                                                             \
	    result = 1;                                                 \
	    break;                                                      \
	  }                                                             \
      }                                                                 \
									\
    if (speed_option_verbose)                                           \
      {                                                                 \
	if (result)                                                     \
	  printf ("%s is microsecond accurate\n", name);                \
	else                                                            \
	  printf ("%s is only %s clock tick accurate\n",                \
		  name, unittime_string (1.0/clk_tck()));               \
      }                                                                 \
    return result;                                                      \
  }


int
gettimeofday_microseconds_p (void)
{
#define call_gettimeofday(t)   gettimeofday (&(t), NULL)
#define timeval_tv_sec(t)      ((t).tv_sec)
#define timeval_tv_usec(t)     ((t).tv_usec)
  MICROSECONDS_P ("gettimeofday", struct_timeval,
		  call_gettimeofday, timeval_tv_sec, timeval_tv_usec);
}

int
getrusage_microseconds_p (void)
{
#define call_getrusage(t)   getrusage (0, &(t))
#define rusage_tv_sec(t)    ((t).ru_utime.tv_sec)
#define rusage_tv_usec(t)   ((t).ru_utime.tv_usec)
  MICROSECONDS_P ("getrusage", struct_rusage,
		  call_getrusage, rusage_tv_sec, rusage_tv_usec);
}

/* Test whether getrusage goes backwards, return non-zero if it does
   (suggesting it's flawed).

   On a macintosh m68040-unknown-netbsd1.4.1 getrusage looks like it's
   microsecond accurate, but has been seen remaining unchanged after many
   microseconds have elapsed.  It also regularly goes backwards by 1000 to
   5000 usecs, this has been seen after between 500 and 4000 attempts taking
   perhaps 0.03 seconds.  We consider this too broken for good measuring.
   We used to have configure pretend getrusage didn't exist on this system,
   but a runtime test should be more reliable, since we imagine the problem
   is not confined to just this exact system tuple.  */

int
getrusage_backwards_p (void)
{
  static int result = -1;
  struct rusage  start, prev, next;
  long  d;
  int   i;

  if (result != -1)
    return result;

  getrusage (0, &start);
  memcpy (&next, &start, sizeof (next));

  result = 0;
  i = 0;
  for (;;)
    {
      memcpy (&prev, &next, sizeof (prev));
      getrusage (0, &next);

      if (next.ru_utime.tv_sec < prev.ru_utime.tv_sec
	  || (next.ru_utime.tv_sec == prev.ru_utime.tv_sec
	      && next.ru_utime.tv_usec < prev.ru_utime.tv_usec))
	{
	  if (speed_option_verbose)
	    printf ("getrusage went backwards (attempt %d: %ld.%06ld -> %ld.%06ld)\n",
		    i,
		    (long) prev.ru_utime.tv_sec, (long) prev.ru_utime.tv_usec,
		    (long) next.ru_utime.tv_sec, (long) next.ru_utime.tv_usec);
	  result = 1;
	  break;
	}

      /* minimum 1000 attempts, then stop after either 0.1 seconds or 50000
	 attempts, whichever comes first */
      d = 1000000 * (next.ru_utime.tv_sec - start.ru_utime.tv_sec)
	+ (next.ru_utime.tv_usec - start.ru_utime.tv_usec);
      i++;
      if (i > 50000 || (i > 1000 && d > 100000))
	break;
    }

  return result;
}

/* CLOCK_PROCESS_CPUTIME_ID looks like it's going to be in a future version
   of glibc (some time post 2.2).

   CLOCK_VIRTUAL is process time, available in BSD systems (though sometimes
   defined, but returning -1 for an error).  */

#ifdef CLOCK_PROCESS_CPUTIME_ID
# define CGT_ID        CLOCK_PROCESS_CPUTIME_ID
#else
# ifdef CLOCK_VIRTUAL
#  define CGT_ID       CLOCK_VIRTUAL
# endif
#endif
#ifdef CGT_ID
const int  have_cgt_id = 1;
#else
const int  have_cgt_id = 0;
# define CGT_ID       (ASSERT_FAIL (CGT_ID not determined), -1)
#endif

#define CGT_DELAY_COUNT 1000

int
cgt_works_p (void)
{
  static int  result = -1;
  struct_timespec  unit;

  if (! have_cgt)
    return 0;

  if (! have_cgt_id)
    {
      if (speed_option_verbose)
	printf ("clock_gettime don't know what ID to use\n");
      result = 0;
      return result;
    }

  if (result != -1)
    return result;

  /* trial run to see if it works */
  if (clock_gettime (CGT_ID, &unit) != 0)
    {
      if (speed_option_verbose)
	printf ("clock_gettime id=%d error: %s\n", CGT_ID, strerror (errno));
      result = 0;
      return result;
    }

  /* get the resolution */
  if (clock_getres (CGT_ID, &unit) != 0)
    {
      if (speed_option_verbose)
	printf ("clock_getres id=%d error: %s\n", CGT_ID, strerror (errno));
      result = 0;
      return result;
    }

  cgt_unittime = unit.tv_sec + unit.tv_nsec * 1e-9;
  printf ("clock_gettime is %s accurate\n",
	  unittime_string (cgt_unittime));

  if (cgt_unittime < 10e-9)
    {
      /* Do we believe this? */
      struct timespec start, end;
      static volatile int counter;
      double duration;
      if (clock_gettime (CGT_ID, &start))
	{
	  if (speed_option_verbose)
	    printf ("clock_gettime id=%d error: %s\n", CGT_ID, strerror (errno));
	  result = 0;
	  return result;
	}
      /* Loop of at least 1000 memory accesses, ought to take at
	 least 100 ns*/
      for (counter = 0; counter < CGT_DELAY_COUNT; counter++)
	;
      if (clock_gettime (CGT_ID, &end))
	{
	  if (speed_option_verbose)
	    printf ("clock_gettime id=%d error: %s\n", CGT_ID, strerror (errno));
	  result = 0;
	  return result;
	}
      duration = (end.tv_sec + end.tv_nsec * 1e-9
		  - start.tv_sec - start.tv_nsec * 1e-9);
      if (speed_option_verbose)
	printf ("delay loop of %d rounds took %s (according to clock_gettime)\n",
		CGT_DELAY_COUNT, unittime_string (duration));
      if (duration < 100e-9)
	{
	  if (speed_option_verbose)
	    printf ("clock_gettime id=%d not believable\n", CGT_ID);
	  result = 0;
	  return result;
	}
    }
  result = 1;
  return result;
}


static double
freq_measure_mftb_one (void)
{
#define call_gettimeofday(t)   gettimeofday (&(t), NULL)
#define timeval_tv_sec(t)      ((t).tv_sec)
#define timeval_tv_usec(t)     ((t).tv_usec)
  FREQ_MEASURE_ONE ("mftb", struct_timeval,
		    call_gettimeofday, MFTB,
		    timeval_tv_sec, timeval_tv_usec);
}


static jmp_buf  mftb_works_buf;

static RETSIGTYPE
mftb_works_handler (int sig)
{
  longjmp (mftb_works_buf, 1);
}

int
mftb_works_p (void)
{
  unsigned   a[2];
  RETSIGTYPE (*old_handler) (int);
  double     cycletime;

  /* suppress a warning about a[] unused */
  a[0] = 0;

  if (! have_mftb)
    return 0;

#ifdef SIGILL
  old_handler = signal (SIGILL, mftb_works_handler);
  if (old_handler == SIG_ERR)
    {
      if (speed_option_verbose)
	printf ("mftb_works_p(): SIGILL not supported, assuming mftb works\n");
      return 1;
    }
  if (setjmp (mftb_works_buf))
    {
      if (speed_option_verbose)
	printf ("mftb_works_p(): SIGILL during mftb, so doesn't work\n");
      return 0;
    }
  MFTB (a);
  signal (SIGILL, old_handler);
  if (speed_option_verbose)
    printf ("mftb_works_p(): mftb works\n");
#else

  if (speed_option_verbose)
    printf ("mftb_works_p(): SIGILL not defined, assuming mftb works\n");
#endif

#if ! HAVE_GETTIMEOFDAY
  if (speed_option_verbose)
    printf ("mftb_works_p(): no gettimeofday available to measure mftb\n");
  return 0;
#endif

  /* The time base is normally 1/4 of the bus speed on 6xx and 7xx chips, on
     other chips it can be driven from an external clock. */
  cycletime = freq_measure ("mftb", freq_measure_mftb_one);
  if (cycletime == -1.0)
    {
      if (speed_option_verbose)
	printf ("mftb_works_p(): cannot measure mftb period\n");
      return 0;
    }

  mftb_unittime = cycletime;
  return 1;
}


volatile unsigned  *sgi_addr;

int
sgi_works_p (void)
{
#if HAVE_SYSSGI && HAVE_MMAP
  static int  result = -1;

  size_t          pagesize, offset;
  __psunsigned_t  phys, physpage;
  void            *virtpage;
  unsigned        period_picoseconds;
  int             size, fd;

  if (result != -1)
    return result;

  phys = syssgi (SGI_QUERY_CYCLECNTR, &period_picoseconds);
  if (phys == (__psunsigned_t) -1)
    {
      /* ENODEV is the error when a counter is not available */
      if (speed_option_verbose)
	printf ("syssgi SGI_QUERY_CYCLECNTR error: %s\n", strerror (errno));
      result = 0;
      return result;
    }
  sgi_unittime = period_picoseconds * 1e-12;

  /* IRIX 5 doesn't have SGI_CYCLECNTR_SIZE, assume 32 bits in that case.
     Challenge/ONYX hardware has a 64 bit byte counter, but there seems no
     obvious way to identify that without SGI_CYCLECNTR_SIZE.  */
#ifdef SGI_CYCLECNTR_SIZE
  size = syssgi (SGI_CYCLECNTR_SIZE);
  if (size == -1)
    {
      if (speed_option_verbose)
	{
	  printf ("syssgi SGI_CYCLECNTR_SIZE error: %s\n", strerror (errno));
	  printf ("    will assume size==4\n");
	}
      size = 32;
    }
#else
  size = 32;
#endif

  if (size < 32)
    {
      printf ("syssgi SGI_CYCLECNTR_SIZE gives %d, expected 32 or 64\n", size);
      result = 0;
      return result;
    }

  pagesize = getpagesize();
  offset = (size_t) phys & (pagesize-1);
  physpage = phys - offset;

  /* shouldn't cross over a page boundary */
  ASSERT_ALWAYS (offset + size/8 <= pagesize);

  fd = open("/dev/mmem", O_RDONLY);
  if (fd == -1)
    {
      if (speed_option_verbose)
	printf ("open /dev/mmem: %s\n", strerror (errno));
      result = 0;
      return result;
    }

  virtpage = mmap (0, pagesize, PROT_READ, MAP_PRIVATE, fd, (off_t) physpage);
  if (virtpage == (void *) -1)
    {
      if (speed_option_verbose)
	printf ("mmap /dev/mmem: %s\n", strerror (errno));
      result = 0;
      return result;
    }

  /* address of least significant 4 bytes, knowing mips is big endian */
  sgi_addr = (unsigned *) ((char *) virtpage + offset
			   + size/8 - sizeof(unsigned));
  result = 1;
  return result;

#else /* ! (HAVE_SYSSGI && HAVE_MMAP) */
  return 0;
#endif
}


#define DEFAULT(var,n)  \
  do {                  \
    if (! (var))        \
      (var) = (n);      \
  } while (0)

void
speed_time_init (void)
{
  double supplement_unittime = 0.0;

  static int  speed_time_initialized = 0;
  if (speed_time_initialized)
    return;
  speed_time_initialized = 1;

  speed_cycletime_init ();

  if (!speed_option_cycles_broken && have_cycles && cycles_works_p ())
    {
      use_cycles = 1;
      DEFAULT (speed_cycletime, 1.0);
      speed_unittime = speed_cycletime;
      DEFAULT (speed_precision, 10000);
      strcpy (speed_time_string, "CPU cycle counter");

      /* only used if a supplementary method is chosen below */
      cycles_limit = (have_cycles == 1 ? M_2POW32 : M_2POW64) / 2.0
	* speed_cycletime;

      if (have_grus && getrusage_microseconds_p() && ! getrusage_backwards_p())
	{
	  /* this is a good combination */
	  use_grus = 1;
	  supplement_unittime = grus_unittime = 1.0e-6;
	  strcpy (speed_time_string, "CPU cycle counter, supplemented by microsecond getrusage()");
	}
      else if (have_cycles == 1)
	{
	  /* When speed_cyclecounter has a limited range, look for something
	     to supplement it. */
	  if (have_gtod && gettimeofday_microseconds_p())
	    {
	      use_gtod = 1;
	      supplement_unittime = gtod_unittime = 1.0e-6;
	      strcpy (speed_time_string, "CPU cycle counter, supplemented by microsecond gettimeofday()");
	    }
	  else if (have_grus)
	    {
	      use_grus = 1;
	      supplement_unittime = grus_unittime = 1.0 / (double) clk_tck ();
	      sprintf (speed_time_string, "CPU cycle counter, supplemented by %s clock tick getrusage()", unittime_string (supplement_unittime));
	    }
	  else if (have_times)
	    {
	      use_times = 1;
	      supplement_unittime = times_unittime = 1.0 / (double) clk_tck ();
	      sprintf (speed_time_string, "CPU cycle counter, supplemented by %s clock tick times()", unittime_string (supplement_unittime));
	    }
	  else if (have_gtod)
	    {
	      use_gtod = 1;
	      supplement_unittime = gtod_unittime = 1.0 / (double) clk_tck ();
	      sprintf (speed_time_string, "CPU cycle counter, supplemented by %s clock tick gettimeofday()", unittime_string (supplement_unittime));
	    }
	  else
	    {
	      fprintf (stderr, "WARNING: cycle counter is 32 bits and there's no other functions.\n");
	      fprintf (stderr, "    Wraparounds may produce bad results on long measurements.\n");
	    }
	}

      if (use_grus || use_times || use_gtod)
	{
	  /* must know cycle period to compare cycles to other measuring
	     (via cycles_limit) */
	  speed_cycletime_need_seconds ();

	  if (speed_precision * supplement_unittime > cycles_limit)
	    {
	      fprintf (stderr, "WARNING: requested precision can't always be achieved due to limited range\n");
	      fprintf (stderr, "    cycle counter and limited precision supplemental method\n");
	      fprintf (stderr, "    (%s)\n", speed_time_string);
	    }
	}
    }
  else if (have_stck)
    {
      strcpy (speed_time_string, "STCK timestamp");
      /* stck is in units of 2^-12 microseconds, which is very likely higher
	 resolution than a cpu cycle */
      if (speed_cycletime == 0.0)
	speed_cycletime_fail
	  ("Need to know CPU frequency for effective stck unit");
      speed_unittime = MAX (speed_cycletime, STCK_PERIOD);
      DEFAULT (speed_precision, 10000);
    }
  else if (have_mftb && mftb_works_p ())
    {
      use_mftb = 1;
      DEFAULT (speed_precision, 10000);
      speed_unittime = mftb_unittime;
      sprintf (speed_time_string, "mftb counter (%s)",
	       unittime_string (speed_unittime));
    }
  else if (have_sgi && sgi_works_p ())
    {
      use_sgi = 1;
      DEFAULT (speed_precision, 10000);
      speed_unittime = sgi_unittime;
      sprintf (speed_time_string, "syssgi() mmap counter (%s), supplemented by millisecond getrusage()",
	       unittime_string (speed_unittime));
      /* supplemented with getrusage, which we assume to have 1ms resolution */
      use_grus = 1;
      supplement_unittime = 1e-3;
    }
  else if (have_rrt)
    {
      timebasestruct_t  t;
      use_rrt = 1;
      DEFAULT (speed_precision, 10000);
      read_real_time (&t, sizeof(t));
      switch (t.flag) {
      case RTC_POWER:
	/* FIXME: What's the actual RTC resolution? */
	speed_unittime = 1e-7;
	strcpy (speed_time_string, "read_real_time() power nanoseconds");
	break;
      case RTC_POWER_PC:
	t.tb_high = 1;
	t.tb_low = 0;
	time_base_to_time (&t, sizeof(t));
	speed_unittime = TIMEBASESTRUCT_SECS(&t) / M_2POW32;
	sprintf (speed_time_string, "%s read_real_time() powerpc ticks",
		 unittime_string (speed_unittime));
	break;
      default:
	fprintf (stderr, "ERROR: Unrecognised timebasestruct_t flag=%d\n",
		 t.flag);
	abort ();
      }
    }
  else if (have_cgt && cgt_works_p() && cgt_unittime < 1.5e-6)
    {
      /* use clock_gettime if microsecond or better resolution */
    choose_cgt:
      use_cgt = 1;
      speed_unittime = cgt_unittime;
      DEFAULT (speed_precision, (cgt_unittime <= 0.1e-6 ? 10000 : 1000));
      strcpy (speed_time_string, "microsecond accurate clock_gettime()");
    }
  else if (have_times && clk_tck() > 1000000)
    {
      /* Cray vector systems have times() which is clock cycle resolution
	 (eg. 450 MHz).  */
      DEFAULT (speed_precision, 10000);
      goto choose_times;
    }
  else if (have_grus && getrusage_microseconds_p() && ! getrusage_backwards_p())
    {
      use_grus = 1;
      speed_unittime = grus_unittime = 1.0e-6;
      DEFAULT (speed_precision, 1000);
      strcpy (speed_time_string, "microsecond accurate getrusage()");
    }
  else if (have_gtod && gettimeofday_microseconds_p())
    {
      use_gtod = 1;
      speed_unittime = gtod_unittime = 1.0e-6;
      DEFAULT (speed_precision, 1000);
      strcpy (speed_time_string, "microsecond accurate gettimeofday()");
    }
  else if (have_cgt && cgt_works_p() && cgt_unittime < 1.5/clk_tck())
    {
      /* use clock_gettime if 1 tick or better resolution */
      goto choose_cgt;
    }
  else if (have_times)
    {
      use_tick_boundary = 1;
      DEFAULT (speed_precision, 200);
    choose_times:
      use_times = 1;
      speed_unittime = times_unittime = 1.0 / (double) clk_tck ();
      sprintf (speed_time_string, "%s clock tick times()",
	       unittime_string (speed_unittime));
    }
  else if (have_grus)
    {
      use_grus = 1;
      use_tick_boundary = 1;
      speed_unittime = grus_unittime = 1.0 / (double) clk_tck ();
      DEFAULT (speed_precision, 200);
      sprintf (speed_time_string, "%s clock tick getrusage()\n",
	       unittime_string (speed_unittime));
    }
  else if (have_gtod)
    {
      use_gtod = 1;
      use_tick_boundary = 1;
      speed_unittime = gtod_unittime = 1.0 / (double) clk_tck ();
      DEFAULT (speed_precision, 200);
      sprintf (speed_time_string, "%s clock tick gettimeofday()",
	       unittime_string (speed_unittime));
    }
  else
    {
      fprintf (stderr, "No time measuring method available\n");
      fprintf (stderr, "None of: speed_cyclecounter(), STCK(), getrusage(), gettimeofday(), times()\n");
      abort ();
    }

  if (speed_option_verbose)
    {
      printf ("speed_time_init: %s\n", speed_time_string);
      printf ("    speed_precision     %d\n", speed_precision);
      printf ("    speed_unittime      %.2g\n", speed_unittime);
      if (supplement_unittime)
	printf ("    supplement_unittime %.2g\n", supplement_unittime);
      printf ("    use_tick_boundary   %d\n", use_tick_boundary);
      if (have_cycles)
	printf ("    cycles_limit        %.2g seconds\n", cycles_limit);
    }
}



/* Burn up CPU until a clock tick boundary, for greater accuracy.  Set the
   corresponding "start_foo" appropriately too. */

void
grus_tick_boundary (void)
{
  struct_rusage  prev;
  getrusage (0, &prev);
  do {
    getrusage (0, &start_grus);
  } while (start_grus.ru_utime.tv_usec == prev.ru_utime.tv_usec);
}

void
gtod_tick_boundary (void)
{
  struct_timeval  prev;
  gettimeofday (&prev, NULL);
  do {
    gettimeofday (&start_gtod, NULL);
  } while (start_gtod.tv_usec == prev.tv_usec);
}

void
times_tick_boundary (void)
{
  struct_tms  prev;
  times (&prev);
  do
    times (&start_times);
  while (start_times.tms_utime == prev.tms_utime);
}


/* "have_" values are tested to let unused code go dead.  */

void
speed_starttime (void)
{
  speed_time_init ();

  if (have_grus && use_grus)
    {
      if (use_tick_boundary)
	grus_tick_boundary ();
      else
	getrusage (0, &start_grus);
    }

  if (have_gtod && use_gtod)
    {
      if (use_tick_boundary)
	gtod_tick_boundary ();
      else
	gettimeofday (&start_gtod, NULL);
    }

  if (have_times && use_times)
    {
      if (use_tick_boundary)
	times_tick_boundary ();
      else
	times (&start_times);
    }

  if (have_cgt && use_cgt)
    clock_gettime (CGT_ID, &start_cgt);

  if (have_rrt && use_rrt)
    read_real_time (&start_rrt, sizeof(start_rrt));

  if (have_sgi && use_sgi)
    start_sgi = *sgi_addr;

  if (have_mftb && use_mftb)
    MFTB (start_mftb);

  if (have_stck && use_stck)
    STCK (start_stck);

  /* Cycles sampled last for maximum accuracy. */
  if (have_cycles && use_cycles)
    speed_cyclecounter (start_cycles);
}


/* Calculate the difference between two cycle counter samples, as a "double"
   counter of cycles.

   The start and end values are allowed to cancel in integers in case the
   counter values are bigger than the 53 bits that normally fit in a double.

   This works even if speed_cyclecounter() puts a value bigger than 32-bits
   in the low word (the high word always gets a 2**32 multiplier though). */

double
speed_cyclecounter_diff (const unsigned end[2], const unsigned start[2])
{
  unsigned  d;
  double    t;

  if (have_cycles == 1)
    {
      t = (end[0] - start[0]);
    }
  else
    {
      d = end[0] - start[0];
      t = d - (d > end[0] ? M_2POWU : 0.0);
      t += (end[1] - start[1]) * M_2POW32;
    }
  return t;
}


double
speed_mftb_diff (const unsigned end[2], const unsigned start[2])
{
  unsigned  d;
  double    t;

  d = end[0] - start[0];
  t = (double) d - (d > end[0] ? M_2POW32 : 0.0);
  t += (end[1] - start[1]) * M_2POW32;
  return t;
}


/* Calculate the difference between "start" and "end" using fields "sec" and
   "psec", where each "psec" is a "punit" of a second.

   The seconds parts are allowed to cancel before being combined with the
   psec parts, in case a simple "sec+psec*punit" exceeds the precision of a
   double.

   Total time is only calculated in a "double" since an integer count of
   psecs might overflow.  2^32 microseconds is only a bit over an hour, or
   2^32 nanoseconds only about 4 seconds.

   The casts to "long" are for the benefit of timebasestruct_t, where the
   fields are only "unsigned int", but we want a signed difference.  */

#define DIFF_SECS_ROUTINE(sec, psec, punit)                     \
  {                                                             \
    long  sec_diff, psec_diff;                                  \
    sec_diff = (long) end->sec - (long) start->sec;             \
    psec_diff = (long) end->psec - (long) start->psec;          \
    return (double) sec_diff + punit * (double) psec_diff;      \
  }

double
timeval_diff_secs (const struct_timeval *end, const struct_timeval *start)
{
  DIFF_SECS_ROUTINE (tv_sec, tv_usec, 1e-6);
}

double
rusage_diff_secs (const struct_rusage *end, const struct_rusage *start)
{
  DIFF_SECS_ROUTINE (ru_utime.tv_sec, ru_utime.tv_usec, 1e-6);
}

double
timespec_diff_secs (const struct_timespec *end, const struct_timespec *start)
{
  DIFF_SECS_ROUTINE (tv_sec, tv_nsec, 1e-9);
}

/* This is for use after time_base_to_time, ie. for seconds and nanoseconds. */
double
timebasestruct_diff_secs (const timebasestruct_t *end,
			  const timebasestruct_t *start)
{
  DIFF_SECS_ROUTINE (tb_high, tb_low, 1e-9);
}


double
speed_endtime (void)
{
#define END_USE(name,value)                             \
  do {                                                  \
    if (speed_option_verbose >= 3)                      \
      printf ("speed_endtime(): used %s\n", name);      \
    result = value;                                     \
    goto done;                                          \
  } while (0)

#define END_ENOUGH(name,value)                                          \
  do {                                                                  \
    if (speed_option_verbose >= 3)                                      \
      printf ("speed_endtime(): %s gives enough precision\n", name);    \
    result = value;                                                     \
    goto done;                                                          \
  } while (0)

#define END_EXCEED(name,value)                                            \
  do {                                                                    \
    if (speed_option_verbose >= 3)                                        \
      printf ("speed_endtime(): cycle counter limit exceeded, used %s\n", \
	      name);                                                      \
    result = value;                                                       \
    goto done;                                                            \
  } while (0)

  unsigned          end_cycles[2];
  stck_t            end_stck;
  unsigned          end_mftb[2];
  unsigned          end_sgi;
  timebasestruct_t  end_rrt;
  struct_timespec   end_cgt;
  struct_timeval    end_gtod;
  struct_rusage     end_grus;
  struct_tms        end_times;
  double            t_gtod, t_grus, t_times, t_cgt;
  double            t_rrt, t_sgi, t_mftb, t_stck, t_cycles;
  double            result;

  /* Cycles sampled first for maximum accuracy.
     "have_" values tested to let unused code go dead.  */

  if (have_cycles && use_cycles)  speed_cyclecounter (end_cycles);
  if (have_stck   && use_stck)    STCK (end_stck);
  if (have_mftb   && use_mftb)    MFTB (end_mftb);
  if (have_sgi    && use_sgi)     end_sgi = *sgi_addr;
  if (have_rrt    && use_rrt)     read_real_time (&end_rrt, sizeof(end_rrt));
  if (have_cgt    && use_cgt)     clock_gettime (CGT_ID, &end_cgt);
  if (have_gtod   && use_gtod)    gettimeofday (&end_gtod, NULL);
  if (have_grus   && use_grus)    getrusage (0, &end_grus);
  if (have_times  && use_times)   times (&end_times);

  result = -1.0;

  if (speed_option_verbose >= 4)
    {
      printf ("speed_endtime():\n");
      if (use_cycles)
	printf ("   cycles  0x%X,0x%X -> 0x%X,0x%X\n",
		start_cycles[1], start_cycles[0],
		end_cycles[1], end_cycles[0]);

      if (use_stck)
	printf ("   stck  0x%lX -> 0x%lX\n", start_stck, end_stck);

      if (use_mftb)
	printf ("   mftb  0x%X,%08X -> 0x%X,%08X\n",
		start_mftb[1], start_mftb[0],
		end_mftb[1], end_mftb[0]);

      if (use_sgi)
	printf ("   sgi  0x%X -> 0x%X\n", start_sgi, end_sgi);

      if (use_rrt)
	printf ("   read_real_time  (%d)%u,%u -> (%d)%u,%u\n",
		start_rrt.flag, start_rrt.tb_high, start_rrt.tb_low,
		end_rrt.flag, end_rrt.tb_high, end_rrt.tb_low);

      if (use_cgt)
	printf ("   clock_gettime  %ld.%09ld -> %ld.%09ld\n",
		start_cgt.tv_sec, start_cgt.tv_nsec,
		end_cgt.tv_sec, end_cgt.tv_nsec);

      if (use_gtod)
	printf ("   gettimeofday  %ld.%06ld -> %ld.%06ld\n",
		start_gtod.tv_sec, start_gtod.tv_usec,
		end_gtod.tv_sec, end_gtod.tv_usec);

      if (use_grus)
	printf ("   getrusage  %ld.%06ld -> %ld.%06ld\n",
		start_grus.ru_utime.tv_sec, start_grus.ru_utime.tv_usec,
		end_grus.ru_utime.tv_sec, end_grus.ru_utime.tv_usec);

      if (use_times)
	printf ("   times  %ld -> %ld\n",
		start_times.tms_utime, end_times.tms_utime);
    }

  if (use_rrt)
    {
      time_base_to_time (&start_rrt, sizeof(start_rrt));
      time_base_to_time (&end_rrt, sizeof(end_rrt));
      t_rrt = timebasestruct_diff_secs (&end_rrt, &start_rrt);
      END_USE ("read_real_time()", t_rrt);
    }

  if (use_cgt)
    {
      t_cgt = timespec_diff_secs (&end_cgt, &start_cgt);
      END_USE ("clock_gettime()", t_cgt);
    }

  if (use_grus)
    {
      t_grus = rusage_diff_secs (&end_grus, &start_grus);

      /* Use getrusage() if the cycle counter limit would be exceeded, or if
	 it provides enough accuracy already. */
      if (use_cycles)
	{
	  if (t_grus >= speed_precision*grus_unittime)
	    END_ENOUGH ("getrusage()", t_grus);
	  if (t_grus >= cycles_limit)
	    END_EXCEED ("getrusage()", t_grus);
	}
    }

  if (use_times)
    {
      t_times = (end_times.tms_utime - start_times.tms_utime) * times_unittime;

      /* Use times() if the cycle counter limit would be exceeded, or if
	 it provides enough accuracy already. */
      if (use_cycles)
	{
	  if (t_times >= speed_precision*times_unittime)
	    END_ENOUGH ("times()", t_times);
	  if (t_times >= cycles_limit)
	    END_EXCEED ("times()", t_times);
	}
    }

  if (use_gtod)
    {
      t_gtod = timeval_diff_secs (&end_gtod, &start_gtod);

      /* Use gettimeofday() if it measured a value bigger than the cycle
	 counter can handle.  */
      if (use_cycles)
	{
	  if (t_gtod >= cycles_limit)
	    END_EXCEED ("gettimeofday()", t_gtod);
	}
    }

  if (use_mftb)
    {
      t_mftb = speed_mftb_diff (end_mftb, start_mftb) * mftb_unittime;
      END_USE ("mftb", t_mftb);
    }

  if (use_stck)
    {
      t_stck = (end_stck - start_stck) * STCK_PERIOD;
      END_USE ("stck", t_stck);
    }

  if (use_sgi)
    {
      t_sgi = (end_sgi - start_sgi) * sgi_unittime;
      END_USE ("SGI hardware counter", t_sgi);
    }

  if (use_cycles)
    {
      t_cycles = speed_cyclecounter_diff (end_cycles, start_cycles)
	* speed_cycletime;
      END_USE ("cycle counter", t_cycles);
    }

  if (use_grus && getrusage_microseconds_p())
    END_USE ("getrusage()", t_grus);

  if (use_gtod && gettimeofday_microseconds_p())
    END_USE ("gettimeofday()", t_gtod);

  if (use_times)  END_USE ("times()",        t_times);
  if (use_grus)   END_USE ("getrusage()",    t_grus);
  if (use_gtod)   END_USE ("gettimeofday()", t_gtod);

  fprintf (stderr, "speed_endtime(): oops, no time method available\n");
  abort ();

 done:
  if (result < 0.0)
    {
      if (speed_option_verbose >= 2)
	fprintf (stderr, "speed_endtime(): warning, treating negative time as zero: %.9f\n", result);
      result = 0.0;
    }
  return result;
}
