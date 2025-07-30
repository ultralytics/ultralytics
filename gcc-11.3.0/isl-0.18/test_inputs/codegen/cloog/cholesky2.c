{
  for (int c1 = 1; c1 <= M; c1 += 1) {
    S1(c1);
    for (int c2 = c1 + 1; c2 <= M; c2 += 1)
      S4(c1, c2);
  }
  for (int c0 = 1; c0 < 3 * M - 1; c0 += 3) {
    S3((c0 + 2) / 3);
    for (int c1 = (c0 + 5) / 3; c1 <= M; c1 += 1) {
      S6((c0 + 2) / 3, c1);
      for (int c4 = (c0 + 5) / 3; c4 < c1; c4 += 1)
        S5(c4, c1, (c0 + 2) / 3);
    }
    for (int c1 = (c0 + 5) / 3; c1 <= M; c1 += 1)
      S2(c1, (c0 + 2) / 3);
  }
}
