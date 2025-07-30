for (int c0 = 0; c0 <= 10; c0 += 1)
  for (int c1 = max(2 * c0 - 4, c0); c1 <= min(2 * c0, c0 + 6); c1 += 1)
    s0(2 * c0 - c1, -c0 + c1);
