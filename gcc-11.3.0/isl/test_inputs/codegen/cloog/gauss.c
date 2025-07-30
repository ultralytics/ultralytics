for (int c0 = 1; c0 < M; c0 += 1)
  for (int c1 = c0 + 1; c1 <= M; c1 += 1) {
    for (int c3 = 1; c3 < c0; c3 += 1)
      S1(c0, c3, c1);
    for (int c3 = c0 + 1; c3 <= M; c3 += 1)
      S2(c0, c3, c1);
  }
