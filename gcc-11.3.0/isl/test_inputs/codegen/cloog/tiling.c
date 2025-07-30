for (int c0 = 0; c0 <= n / 10; c0 += 1)
  for (int c1 = 10 * c0; c1 <= min(n, 10 * c0 + 9); c1 += 1)
    S1(c0, c1);
