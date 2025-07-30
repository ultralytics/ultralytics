{
  S3(1, 0);
  for (int c2 = 2; c2 <= M; c2 += 1)
    S1(1, 1, c2);
  for (int c0 = 2; c0 <= M; c0 += 1) {
    S4(c0, 0);
    for (int c2 = c0 + 1; c2 <= M; c2 += 1)
      S2(c0, 1, c2);
  }
}
