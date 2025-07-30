/* CPU frequency determination.

Copyright 1999-2004 Free Software Foundation, Inc.

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


/* Currently we don't get a CPU frequency on the following systems,

   alphaev5-cray-unicosmk2.0.6.X
       times() has been seen at 13.33 ns (75 MHz), which is probably not the
       cpu frequency.  Measuring the cycle counter against that would be
       possible though.  But currently we don't use the cycle counter due to
       unicos having int==8bytes where tune/alpha.asm assumes int==4bytes.

   m68040-unknown-netbsd1.4.1
       Not sure if the system even knows the cpu frequency.  There's no
       cycle counter to measure, though we could perhaps make a loop taking
       a known number of cycles and measure that.

   power-ibm-aix4.2.1.0
   power2-ibm-aix4.3.1.0
   powerpc604-ibm-aix4.3.1.0
   powerpc604-ibm-aix4.3.3.0
   powerpc630-ibm-aix4.3.3.0
   powerpc-unknown-netbsd1.6
       Don't know where any info hides on these.  mftb is not related to the
       cpu frequency so doesn't help.

   sparc-unknown-linux-gnu [maybe]
       Don't know where any info hides on this.

   t90-cray-unicos10.0.X
       The times() call seems to be for instance 2.22 nanoseconds, which
       might be the cpu frequency (450 mhz), but need to confirm that.

*/

#include "config.h"

#if HAVE_INVENT_H
#include <invent.h> /* for IRIX invent_cpuinfo_t */
#endif

#include <stdio.h>
#include <stdlib.h> /* for getenv, qsort */
#include <string.h> /* for memcmp */

#if HAVE_UNISTD_H
#include <unistd.h> /* for sysconf */
#endif

#include <sys/types.h>

#if HAVE_SYS_ATTRIBUTES_H
#include <sys/attributes.h>   /* for IRIX attr_get(), needs sys/types.h */
#endif

#if HAVE_SYS_IOGRAPH_H
#include <sys/iograph.h>      /* for IRIX INFO_LBL_DETAIL_INVENT */
#endif

#if HAVE_SYS_PARAM_H     /* for constants needed by NetBSD <sys/sysctl.h> */
#include <sys/param.h>   /* and needed by HPUX <sys/pstat.h> */
#endif

#if HAVE_SYS_PSTAT_H
#include <sys/pstat.h>   /* for HPUX pstat_getprocessor() */
#endif

#if HAVE_SYS_SYSCTL_H
#include <sys/sysctl.h>  /* for sysctlbyname() */
#endif

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

#if HAVE_SYS_RESOURCE_H
#include <sys/resource.h>  /* for struct rusage */
#endif

#if HAVE_SYS_PROCESSOR_H
#include <sys/processor.h>  /* for solaris processor_info_t */
#endif

/* On AIX 5.1 with gcc 2.9-aix51-020209 in -maix64 mode, <sys/sysinfo.h>
   gets an error about "fill" in "struct cpuinfo" having a negative size,
   apparently due to __64BIT_KERNEL not being defined because _KERNEL is not
   defined.  Avoid this file if we don't actually need it, which we don't on
   AIX since there's no getsysinfo there.  */
#if HAVE_SYS_SYSINFO_H && HAVE_GETSYSINFO
#include <sys/sysinfo.h>  /* for OSF getsysinfo */
#endif

#if HAVE_MACHINE_HAL_SYSINFO_H
#include <machine/hal_sysinfo.h>  /* for OSF GSI_CPU_INFO, struct cpu_info */
#endif

/* Remove definitions from NetBSD <sys/param.h>, to avoid conflicts with
   gmp-impl.h. */
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif

#include "gmp.h"
#include "gmp-impl.h"

#include "speed.h"


#define HELP(str)                       \
  if (help)                             \
    {                                   \
      printf ("    - %s\n", str);       \
      return 0;                         \
    }


/* GMP_CPU_FREQUENCY environment variable.  Should be in Hertz and can be
   floating point, for example "450e6". */
static int
freq_environment (int help)
{
  char  *e;

  HELP ("environment variable GMP_CPU_FREQUENCY (in Hertz)");

  e = getenv ("GMP_CPU_FREQUENCY");
  if (e == NULL)
    return 0;

  speed_cycletime = 1.0 / atof (e);

  if (speed_option_verbose)
    printf ("Using GMP_CPU_FREQUENCY %.2f for cycle time %.3g\n",
            atof (e), speed_cycletime);

  return 1;
}


/* getsysinfo is available on OSF, or 4.0 and up at least.
   The man page (on 4.0) suggests a 0 return indicates information not
   available, but that seems to be the normal return for GSI_CPU_INFO.  */
static int
freq_getsysinfo (int help)
{
#if HAVE_GETSYSINFO
  struct cpu_info  c;
  int              start;

  HELP ("getsysinfo() GSI_CPU_INFO");

  start = 0;
  if (getsysinfo (GSI_CPU_INFO, (caddr_t) &c, sizeof (c),
                  &start, NULL, NULL) != -1)
    {
      speed_cycletime = 1e-6 / (double) c.mhz;
      if (speed_option_verbose)
        printf ("Using getsysinfo() GSI_CPU_INFO %u for cycle time %.3g\n",
                c.mhz, speed_cycletime);
      return 1;
    }
#endif
  return 0;
}


/* In HPUX 10 and up, pstat_getprocessor() psp_iticksperclktick is the
   number of CPU cycles (ie. the CR16 register) per CLK_TCK.  HPUX 9 doesn't
   have that field in pst_processor though, and has no apparent
   equivalent.  */

static int
freq_pstat_getprocessor (int help)
{
#if HAVE_PSTAT_GETPROCESSOR && HAVE_PSP_ITICKSPERCLKTICK
  struct pst_processor  p;

  HELP ("pstat_getprocessor() psp_iticksperclktick");

  if (pstat_getprocessor (&p, sizeof(p), 1, 0) != -1)
    {
      long  c = clk_tck();
      speed_cycletime = 1.0 / (c * p.psp_iticksperclktick);
      if (speed_option_verbose)
        printf ("Using pstat_getprocessor() psp_iticksperclktick %lu and clk_tck %ld for cycle time %.3g\n",
                (unsigned long) p.psp_iticksperclktick, c,
                speed_cycletime);
      return 1;
    }
#endif
  return 0;
}


/* i386 FreeBSD 2.2.8 sysctlbyname machdep.i586_freq is in Hertz.
   There's no obvious defines available to get this from plain sysctl.  */
static int
freq_sysctlbyname_i586_freq (int help)
{
#if HAVE_SYSCTLBYNAME
  unsigned  val;
  size_t    size;

  HELP ("sysctlbyname() machdep.i586_freq");

  size = sizeof(val);
  if (sysctlbyname ("machdep.i586_freq", &val, &size, NULL, 0) == 0
      && size == sizeof(val))
    {
      speed_cycletime = 1.0 / (double) val;
      if (speed_option_verbose)
        printf ("Using sysctlbyname() machdep.i586_freq %u for cycle time %.3g\n",
                val, speed_cycletime);
      return 1;
    }
#endif
  return 0;
}


/* i368 FreeBSD 3.3 sysctlbyname machdep.tsc_freq is in Hertz.
   There's no obvious defines to get this from plain sysctl.  */

static int
freq_sysctlbyname_tsc_freq (int help)
{
#if HAVE_SYSCTLBYNAME
  unsigned  val;
  size_t    size;

  HELP ("sysctlbyname() machdep.tsc_freq");

  size = sizeof(val);
  if (sysctlbyname ("machdep.tsc_freq", &val, &size, NULL, 0) == 0
      && size == sizeof(val))
    {
      speed_cycletime = 1.0 / (double) val;
      if (speed_option_verbose)
        printf ("Using sysctlbyname() machdep.tsc_freq %u for cycle time %.3g\n",
                val, speed_cycletime);
      return 1;
    }
#endif
  return 0;
}


/* Apple powerpc Darwin 1.3 sysctl hw.cpufrequency is in hertz.  For some
   reason only seems to be available from sysctl(), not sysctlbyname().  */

static int
freq_sysctl_hw_cpufrequency (int help)
{
#if HAVE_SYSCTL && defined (CTL_HW) && defined (HW_CPU_FREQ)
  int       mib[2];
  unsigned  val;
  size_t    size;

  HELP ("sysctl() hw.cpufrequency");

  mib[0] = CTL_HW;
  mib[1] = HW_CPU_FREQ;
  size = sizeof(val);
  if (sysctl (mib, 2, &val, &size, NULL, 0) == 0)
    {
      speed_cycletime = 1.0 / (double) val;
      if (speed_option_verbose)
        printf ("Using sysctl() hw.cpufrequency %u for cycle time %.3g\n",
                val, speed_cycletime);
      return 1;
    }
#endif
  return 0;
}


/* The following ssyctl hw.model strings have been observed,

       Alpha FreeBSD 4.1:   Digital AlphaPC 164LX 599 MHz
       NetBSD 1.4:          Digital AlphaPC 164LX 599 MHz
       NetBSD 1.6.1:        CY7C601 @ 40 MHz, TMS390C602A FPU

   NetBSD 1.4 doesn't seem to have sysctlbyname, so sysctl() is used.  */

static int
freq_sysctl_hw_model (int help)
{
#if HAVE_SYSCTL && defined (CTL_HW) && defined (HW_MODEL)
  int       mib[2];
  char      str[128];
  unsigned  val;
  size_t    size;
  char      *p;
  int       end;

  HELP ("sysctl() hw.model");

  mib[0] = CTL_HW;
  mib[1] = HW_MODEL;
  size = sizeof(str);
  if (sysctl (mib, 2, str, &size, NULL, 0) == 0)
    {
      for (p = str; *p != '\0'; p++)
        {
          end = 0;
          if (sscanf (p, "%u MHz%n", &val, &end) == 1 && end != 0)
            {
              speed_cycletime = 1e-6 / (double) val;
              if (speed_option_verbose)
                printf ("Using sysctl() hw.model %u for cycle time %.3g\n",
                        val, speed_cycletime);
              return 1;
            }
        }
    }
#endif
  return 0;
}


/* /proc/cpuinfo for linux kernel.

   Linux doesn't seem to have any system call to get the CPU frequency, at
   least not in 2.0.x or 2.2.x, so it's necessary to read /proc/cpuinfo.

   i386 2.0.36 - "bogomips" is the CPU frequency.

   i386 2.2.13 - has both "cpu MHz" and "bogomips", and it's "cpu MHz" which
                 is the frequency.

   alpha 2.2.5 - "cycle frequency [Hz]" seems to be right, "BogoMIPS" is
                 very slightly different.

   alpha 2.2.18pre21 - "cycle frequency [Hz]" is 0 on at least one system,
                 "BogoMIPS" seems near enough.

   powerpc 2.2.19 - "clock" is the frequency, bogomips is something weird
  */

static int
freq_proc_cpuinfo (int help)
{
  FILE    *fp;
  char    buf[128];
  double  val;
  int     ret = 0;
  int     end;

  HELP ("linux kernel /proc/cpuinfo file, cpu MHz or bogomips");

  if ((fp = fopen ("/proc/cpuinfo", "r")) != NULL)
    {
      while (fgets (buf, sizeof (buf), fp) != NULL)
        {
          if (sscanf (buf, "cycle frequency [Hz]    : %lf", &val) == 1
              && val != 0.0)
            {
              speed_cycletime = 1.0 / val;
              if (speed_option_verbose)
                printf ("Using /proc/cpuinfo \"cycle frequency\" %.2f for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
          if (sscanf (buf, "cpu MHz : %lf\n", &val) == 1)
            {
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using /proc/cpuinfo \"cpu MHz\" %.2f for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
          end = 0;
          if (sscanf (buf, "clock : %lfMHz\n%n", &val, &end) == 1 && end != 0)
            {
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using /proc/cpuinfo \"clock\" %.2f for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
          if (sscanf (buf, "bogomips : %lf\n", &val) == 1
              || sscanf (buf, "BogoMIPS : %lf\n", &val) == 1)
            {
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using /proc/cpuinfo \"bogomips\" %.2f for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
        }
      fclose (fp);
    }
  return ret;
}


/* /bin/sysinfo for SunOS 4.
   Prints a line like: cpu0 is a "75 MHz TI,TMS390Z55" CPU */
static int
freq_sunos_sysinfo (int help)
{
  int     ret = 0;
#if HAVE_POPEN
  FILE    *fp;
  char    buf[128];
  double  val;
  int     end;

  HELP ("SunOS /bin/sysinfo program output, cpu0");

  /* Error messages are sent to /dev/null in case /bin/sysinfo doesn't
     exist.  The brackets are necessary for some shells. */
  if ((fp = popen ("(/bin/sysinfo) 2>/dev/null", "r")) != NULL)
    {
      while (fgets (buf, sizeof (buf), fp) != NULL)
        {
          end = 0;
          if (sscanf (buf, " cpu0 is a \"%lf MHz%n", &val, &end) == 1
              && end != 0)
            {
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using /bin/sysinfo \"cpu0 MHz\" %.2f for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
        }
      pclose (fp);
    }
#endif
  return ret;
}


/* "/etc/hw -r cpu" for SCO OpenUnix 8, printing a line like
	The speed of the CPU is approximately 450MHz
 */
static int
freq_sco_etchw (int help)
{
  int     ret = 0;
#if HAVE_POPEN
  FILE    *fp;
  char    buf[128];
  double  val;
  int     end;

  HELP ("SCO /etc/hw program output");

  /* Error messages are sent to /dev/null in case /etc/hw doesn't exist.
     The brackets are necessary for some shells. */
  if ((fp = popen ("(/etc/hw -r cpu) 2>/dev/null", "r")) != NULL)
    {
      while (fgets (buf, sizeof (buf), fp) != NULL)
        {
          end = 0;
          if (sscanf (buf, " The speed of the CPU is approximately %lfMHz%n",
                      &val, &end) == 1 && end != 0)
            {
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using /etc/hw %.2f MHz, for cycle time %.3g\n",
                        val, speed_cycletime);
              ret = 1;
              break;
            }
        }
      pclose (fp);
    }
#endif
  return ret;
}


/* attr_get("/hw/cpunum/0",INFO_LBL_DETAIL_INVENT) ic_cpu_info.cpufq for
   IRIX 6.5.  Past versions don't have INFO_LBL_DETAIL_INVENT,
   invent_cpuinfo_t, or /hw/cpunum/0.

   The same information is available from the "hinv -c processor" command,
   but it seems better to make a system call where possible. */

static int
freq_attr_get_invent (int help)
{
  int     ret = 0;
#if HAVE_ATTR_GET && HAVE_INVENT_H && defined (INFO_LBL_DETAIL_INVENT)
  invent_cpuinfo_t  inv;
  int               len, val;

  HELP ("attr_get(\"/hw/cpunum/0\") ic_cpu_info.cpufq");

  len = sizeof (inv);
  if (attr_get ("/hw/cpunum/0", INFO_LBL_DETAIL_INVENT,
                (char *) &inv, &len, 0) == 0
      && len == sizeof (inv)
      && inv.ic_gen.ig_invclass == INV_PROCESSOR)
    {
      val = inv.ic_cpu_info.cpufq;
      speed_cycletime = 1e-6 / val;
      if (speed_option_verbose)
        printf ("Using attr_get(\"/hw/cpunum/0\") ic_cpu_info.cpufq %d MHz for cycle time %.3g\n", val, speed_cycletime);
      ret = 1;
    }
#endif
  return ret;
}


/* FreeBSD on i386 gives a line like the following at bootup, and which can
   be read back from /var/run/dmesg.boot.

       CPU: AMD Athlon(tm) Processor (755.29-MHz 686-class CPU)
       CPU: Pentium 4 (1707.56-MHz 686-class CPU)
       CPU: i486 DX4 (486-class CPU)

   This is useful on FreeBSD 4.x, where there's no sysctl machdep.tsc_freq
   or machdep.i586_freq.

   It's better to use /var/run/dmesg.boot than to run /sbin/dmesg, since the
   latter prints the current system message buffer, which is a limited size
   and can wrap around if the system is up for a long time.  */

static int
freq_bsd_dmesg (int help)
{
  FILE    *fp;
  char    buf[256], *p;
  double  val;
  int     ret = 0;
  int     end;

  HELP ("BSD /var/run/dmesg.boot file");

  if ((fp = fopen ("/var/run/dmesg.boot", "r")) != NULL)
    {
      while (fgets (buf, sizeof (buf), fp) != NULL)
        {
          if (memcmp (buf, "CPU:", 4) == 0)
            {
              for (p = buf; *p != '\0'; p++)
                {
                  end = 0;
                  if (sscanf (p, "(%lf-MHz%n", &val, &end) == 1 && end != 0)
                    {
                      speed_cycletime = 1e-6 / val;
                      if (speed_option_verbose)
                        printf ("Using /var/run/dmesg.boot CPU: %.2f MHz for cycle time %.3g\n", val, speed_cycletime);
                      ret = 1;
                      break;
                    }
                }
            }
        }
      fclose (fp);
    }
  return ret;
}


/* "hinv -c processor" for IRIX.  The following lines have been seen,

              1 150 MHZ IP20 Processor
              2 195 MHZ IP27 Processors
              Processor 0: 500 MHZ IP35

   This information is available from attr_get() on IRIX 6.5 (see above),
   but on IRIX 6.2 it's not clear where to look, so fall back on
   parsing.  */

static int
freq_irix_hinv (int help)
{
  int     ret = 0;
#if HAVE_POPEN
  FILE    *fp;
  char    buf[128];
  double  val;
  int     nproc, end;

  HELP ("IRIX \"hinv -c processor\" output");

  /* Error messages are sent to /dev/null in case hinv doesn't exist.  The
     brackets are necessary for some shells. */
  if ((fp = popen ("(hinv -c processor) 2>/dev/null", "r")) != NULL)
    {
      while (fgets (buf, sizeof (buf), fp) != NULL)
        {
          end = 0;
          if (sscanf (buf, "Processor 0: %lf MHZ%n", &val, &end) == 1
              && end != 0)
            {
            found:
              speed_cycletime = 1e-6 / val;
              if (speed_option_verbose)
                printf ("Using hinv -c processor \"%.2f MHZ\" for cycle time %.3g\n", val, speed_cycletime);
              ret = 1;
              break;
            }
          end = 0;
          if (sscanf (buf, "%d %lf MHZ%n", &nproc, &val, &end) == 2
              && end != 0)
            goto found;
        }
      pclose (fp);
    }
#endif
  return ret;
}


/* processor_info() for Solaris.  "psrinfo" is the command-line interface to
   this.  "prtconf -vp" gives similar information.

   Apple Darwin has a processor_info, but in an incompatible style.  It
   doesn't have <sys/processor.h>, so test for that.  */

static int
freq_processor_info (int help)
{
#if HAVE_PROCESSOR_INFO && HAVE_SYS_PROCESSOR_H
  processor_info_t  p;
  int  i, n, mhz = 0;

  HELP ("processor_info() pi_clock");

  n = sysconf (_SC_NPROCESSORS_CONF);
  for (i = 0; i < n; i++)
    {
      if (processor_info (i, &p) != 0)
        continue;
      if (p.pi_state != P_ONLINE)
        continue;

      if (mhz != 0 && p.pi_clock != mhz)
        {
          fprintf (stderr,
                   "freq_processor_info(): There's more than one CPU and they have different clock speeds\n");
          return 0;
        }

      mhz = p.pi_clock;
    }

  speed_cycletime = 1.0e-6 / (double) mhz;

  if (speed_option_verbose)
    printf ("Using processor_info() %d mhz for cycle time %.3g\n",
            mhz, speed_cycletime);
  return 1;

#else
  return 0;
#endif
}


#if HAVE_SPEED_CYCLECOUNTER && HAVE_GETTIMEOFDAY
static double
freq_measure_gettimeofday_one (void)
{
#define call_gettimeofday(t)   gettimeofday (&(t), NULL)
#define timeval_tv_sec(t)      ((t).tv_sec)
#define timeval_tv_usec(t)     ((t).tv_usec)
  FREQ_MEASURE_ONE ("gettimeofday", struct timeval,
                    call_gettimeofday, speed_cyclecounter,
                    timeval_tv_sec, timeval_tv_usec);
}
#endif

#if HAVE_SPEED_CYCLECOUNTER && HAVE_GETRUSAGE
static double
freq_measure_getrusage_one (void)
{
#define call_getrusage(t)   getrusage (0, &(t))
#define rusage_tv_sec(t)    ((t).ru_utime.tv_sec)
#define rusage_tv_usec(t)   ((t).ru_utime.tv_usec)
  FREQ_MEASURE_ONE ("getrusage", struct rusage,
                    call_getrusage, speed_cyclecounter,
                    rusage_tv_sec, rusage_tv_usec);
}
#endif


/* MEASURE_MATCH is how many readings within MEASURE_TOLERANCE of each other
   are required.  This must be at least 2.  */
#define MEASURE_MAX_ATTEMPTS   20
#define MEASURE_TOLERANCE      1.005  /* 0.5% */
#define MEASURE_MATCH          3

double
freq_measure (const char *name, double (*one) (void))
{
  double  t[MEASURE_MAX_ATTEMPTS];
  int     i, j;

  for (i = 0; i < numberof (t); i++)
    {
      t[i] = (*one) ();

      qsort (t, i+1, sizeof(t[0]), (qsort_function_t) double_cmp_ptr);
      if (speed_option_verbose >= 3)
        for (j = 0; j <= i; j++)
          printf ("   t[%d] is %.6g\n", j, t[j]);

      for (j = 0; j+MEASURE_MATCH-1 <= i; j++)
        {
          if (t[j+MEASURE_MATCH-1] <= t[j] * MEASURE_TOLERANCE)
            {
              /* use the average of the range found */
                return (t[j+MEASURE_MATCH-1] + t[j]) / 2.0;
            }
        }
    }
  return -1.0;
}

static int
freq_measure_getrusage (int help)
{
#if HAVE_SPEED_CYCLECOUNTER && HAVE_GETRUSAGE
  double  cycletime;

  if (! getrusage_microseconds_p ())
    return 0;
  if (! cycles_works_p ())
    return 0;

  HELP ("cycle counter measured with microsecond getrusage()");

  cycletime = freq_measure ("getrusage", freq_measure_getrusage_one);
  if (cycletime == -1.0)
    return 0;

  speed_cycletime = cycletime;
  if (speed_option_verbose)
    printf ("Using getrusage() measured cycle counter %.4g (%.2f MHz)\n",
            speed_cycletime, 1e-6/speed_cycletime);
  return 1;

#else
  return 0;
#endif
}

static int
freq_measure_gettimeofday (int help)
{
#if HAVE_SPEED_CYCLECOUNTER && HAVE_GETTIMEOFDAY
  double  cycletime;

  if (! gettimeofday_microseconds_p ())
    return 0;
  if (! cycles_works_p ())
    return 0;

  HELP ("cycle counter measured with microsecond gettimeofday()");

  cycletime = freq_measure ("gettimeofday", freq_measure_gettimeofday_one);
  if (cycletime == -1.0)
    return 0;

  speed_cycletime = cycletime;
  if (speed_option_verbose)
    printf ("Using gettimeofday() measured cycle counter %.4g (%.2f MHz)\n",
            speed_cycletime, 1e-6/speed_cycletime);
  return 1;
#else
  return 0;
#endif
}


/* Each function returns 1 if it succeeds in setting speed_cycletime, or 0
   if not.

   In general system call tests are first since they're fast, then file
   tests, then tests running programs.  Necessary exceptions to this rule
   are noted.  The measuring is last since it's time consuming, and rather
   wasteful of cpu.  */

static int
freq_all (int help)
{
  return
    /* This should be first, so an environment variable can override
       anything the system gives. */
    freq_environment (help)

    || freq_attr_get_invent (help)
    || freq_getsysinfo (help)
    || freq_pstat_getprocessor (help)
    || freq_sysctl_hw_model (help)
    || freq_sysctl_hw_cpufrequency (help)
    || freq_sysctlbyname_i586_freq (help)
    || freq_sysctlbyname_tsc_freq (help)

    /* SCO openunix 8 puts a dummy pi_clock==16 in processor_info, so be
       sure to check /etc/hw before that function. */
    || freq_sco_etchw (help)

    || freq_processor_info (help)
    || freq_proc_cpuinfo (help)
    || freq_bsd_dmesg (help)
    || freq_irix_hinv (help)
    || freq_sunos_sysinfo (help)
    || freq_measure_getrusage (help)
    || freq_measure_gettimeofday (help);
}


void
speed_cycletime_init (void)
{
  static int  attempted = 0;

  if (attempted)
    return;
  attempted = 1;

  if (freq_all (0))
    return;

  if (speed_option_verbose)
    printf ("CPU frequency couldn't be determined\n");
}


void
speed_cycletime_fail (const char *str)
{
  fprintf (stderr, "Measuring with: %s\n", speed_time_string);
  fprintf (stderr, "%s,\n", str);
  fprintf (stderr, "but none of the following are available,\n");
  freq_all (1);
  abort ();
}

/* speed_time_init leaves speed_cycletime set to either 0.0 or 1.0 when the
   CPU frequency is unknown.  0.0 is when the time base is in seconds, so
   that's no good if cycles are wanted.  1.0 is when the time base is in
   cycles, which conversely is no good if seconds are wanted.  */
void
speed_cycletime_need_cycles (void)
{
  speed_time_init ();
  if (speed_cycletime == 0.0)
    speed_cycletime_fail
      ("Need to know CPU frequency to give times in cycles");
}
void
speed_cycletime_need_seconds (void)
{
  speed_time_init ();
  if (speed_cycletime == 1.0)
    speed_cycletime_fail
      ("Need to know CPU frequency to convert cycles to seconds");
}
