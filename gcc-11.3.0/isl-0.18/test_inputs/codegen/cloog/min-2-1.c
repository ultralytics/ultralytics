for (int c0 = 1; c0 <= N; c0 += 1)
  for (int c1 = 0; c1 <= min(min(M, c0), N - c0); c1 += 1)
    for (int c2 = 0; c2 <= min(min(M, c0), N - c0); c2 += 1)
      S1(c0, c1, c2);
