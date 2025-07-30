{
  S3(2, 1);
  S1(3, 1);
  for (int c0 = 4; c0 <= M + 1; c0 += 1) {
    S1(c0, 1);
    for (int c1 = 2; c1 < (c0 + 1) / 2; c1 += 1)
      S2(c0, c1);
    if (c0 % 2 == 0)
      S4(c0, c0 / 2);
  }
  for (int c0 = M + 2; c0 <= 2 * M; c0 += 1) {
    for (int c1 = -M + c0; c1 < (c0 + 1) / 2; c1 += 1)
      S2(c0, c1);
    if (c0 % 2 == 0)
      S4(c0, c0 / 2);
  }
}
