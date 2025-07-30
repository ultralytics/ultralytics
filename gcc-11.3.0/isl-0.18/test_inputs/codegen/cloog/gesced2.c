{
  for (int c0 = 1; c0 <= 4; c0 += 1)
    for (int c1 = 5; c1 < M - 9; c1 += 1)
      S1(c0, c1);
  for (int c0 = 5; c0 < M - 9; c0 += 1) {
    for (int c1 = -c0 + 1; c1 <= 4; c1 += 1)
      S2(c0 + c1, c0);
    for (int c1 = 5; c1 <= min(M - 10, M - c0); c1 += 1) {
      S1(c0, c1);
      S2(c0 + c1, c0);
    }
    for (int c1 = M - c0 + 1; c1 < M - 9; c1 += 1)
      S1(c0, c1);
    for (int c1 = M - 9; c1 <= M - c0; c1 += 1)
      S2(c0 + c1, c0);
  }
  for (int c0 = M - 9; c0 <= M; c0 += 1)
    for (int c1 = 5; c1 < M - 9; c1 += 1)
      S1(c0, c1);
}
