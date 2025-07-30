for (int c0 = 2; c0 <= n + m; c0 += 1)
  for (int c1 = max(1, -m + c0); c1 <= min(n, c0 - 1); c1 += 1)
    S1(c1, c0 - c1);
