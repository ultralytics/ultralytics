for (int c0 = 0; c0 <= 2 * n; c0 += 1)
  for (int c1 = max(0, -n + c0); c1 <= min(n, c0); c1 += 1)
    S1(c1, c0 - c1);
