for (int c0 = 1; c0 <= M; c0 += 1)
  for (int c1 = 1; c1 < 2 * N; c1 += 1) {
    for (int c2 = max(1, -N + c1); c2 < (c1 + 1) / 2; c2 += 1)
      S1(c0, c1 - c2, c2);
    if ((c1 - 1) % 2 == 0)
      S2(c0, (c1 + 1) / 2);
  }
