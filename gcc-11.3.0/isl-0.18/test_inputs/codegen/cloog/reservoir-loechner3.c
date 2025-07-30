for (int c0 = 1; c0 <= M; c0 += 1)
  for (int c1 = 2; c1 <= M + c0; c1 += 1)
    for (int c2 = max(1, -c0 + c1); c2 <= min(M, c1 - 1); c2 += 1)
      S1(c0, c2, c1 - c2);
