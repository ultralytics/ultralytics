for (int c0 = 0; c0 <= 5; c0 += 1)
  for (int c1 = min(4, 2 * c0); c1 <= max(4, 2 * c0); c1 += 1) {
    if (c1 == 2 * c0)
      S1(c0, 2 * c0);
    if (c1 == 4)
      S2(c0, 4);
  }
