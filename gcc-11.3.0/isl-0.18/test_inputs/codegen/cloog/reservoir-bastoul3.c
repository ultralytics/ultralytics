for (int c0 = 3; c0 <= 9; c0 += 1)
  for (int c1 = max(c0 - 6, -(c0 % 2) + 2); c1 <= min(3, c0 - 2); c1 += 2)
    S1(c0, c1, (c0 - c1) / 2);
