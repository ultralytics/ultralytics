for (int c0 = 0; c0 <= n; c0 += 32)
  for (int c1 = c0; c1 <= min(n, c0 + 31); c1 += 1)
    s0(c0, c1);
