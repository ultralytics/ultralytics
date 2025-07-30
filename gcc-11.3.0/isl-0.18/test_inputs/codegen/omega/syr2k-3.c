for (int c0 = 1; c0 <= min(n, 2 * b - 1); c0 += 1)
  for (int c1 = -b + 1; c1 <= b - c0; c1 += 1)
    for (int c2 = max(1, c0 + c1); c2 <= min(n, n + c1); c2 += 1)
      s0(-c0 - c1 + c2 + 1, -c1 + c2, c2);
