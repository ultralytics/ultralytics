{
  for (int c0 = 1; c0 <= 2; c0 += 1)
    for (int c1 = 1; c1 <= M; c1 += 1)
      S1(c0, c1);
  for (int c0 = 3; c0 <= N; c0 += 1) {
    for (int c1 = 1; c1 <= min(M, c0 - 1); c1 += 1)
      S1(c0, c1);
    if (M >= c0) {
      S1(c0, c0);
      S2(c0, c0);
    }
    for (int c1 = c0 + 1; c1 <= M; c1 += 1)
      S1(c0, c1);
    if (c0 >= M + 1)
      S2(c0, c0);
  }
}
