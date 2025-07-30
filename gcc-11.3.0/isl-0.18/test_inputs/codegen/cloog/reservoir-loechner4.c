for (int c0 = 2; c0 <= 2 * M; c0 += 1)
  for (int c1 = 1; c1 <= M; c1 += 1)
    for (int c2 = 1; c2 <= M; c2 += 1)
      for (int c3 = max(1, -M + c0); c3 <= min(M, c0 - 1); c3 += 1)
        S1(c2, c1, c3, c0 - c3);
